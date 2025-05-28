import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm
import wandb
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path

from temporal_lmg import *
from temporal_binding_demo import DEVICE, IGNORE_INDEX
from thbt_averaged_signals import (
    ImprovedBindingField,
    BindingFieldWrapper,
    build_temporal_model
)

from binding_improvements import patch_binding_loss, add_binding_diagnostics, adjust_binding_threshold



def custom_collate_fn(batch):
    """Custom collate function to handle boundaries"""
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    boundaries = [item['boundaries'] for item in batch]  # List of lists

    return {
        'input_ids': input_ids,
        'labels': labels,
        'boundaries': boundaries
    }

class ConversationDataset(Dataset):
    """Dataset that benefits from temporal binding - conversations have natural semantic chunks"""

    def __init__(
            self,
            tokenizer_path: str = "./personachat_bpe_tokenizer.json",
            max_length: int = 512,
            split: str = "train",
            dataset_name: str = "AlekseyKorshuk/persona-chat"
    ):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length

        # Load dataset
        print(f"Loading {dataset_name} {split} split...")
        self.dataset = load_dataset(dataset_name, split=split)

        # Prepare examples
        self.examples = []
        self._prepare_examples()

    def _prepare_examples(self):
        """Convert conversations to training examples"""
        for item in tqdm(self.dataset, desc="Preparing examples"):
            # PersonaChat format: personality + conversation
            if 'personality' in item:
                context = " ".join(item['personality'])
            else:
                context = ""

            if 'utterances' in item:
                for utterance in item['utterances']:
                    history = utterance.get('history', [])

                    # Create input: context + history
                    if context:
                        full_text = f"Personality: {context}\n"
                    else:
                        full_text = ""

                    # Add conversation history
                    for i, turn in enumerate(history):
                        speaker = "Person1" if i % 2 == 0 else "Person2"
                        full_text += f"{speaker}: {turn}\n"

                    # Tokenize
                    encoding = self.tokenizer.encode(full_text)

                    if len(encoding.ids) > 10:  # Skip very short examples
                        self.examples.append({
                            'input_ids': encoding.ids,
                            'text': full_text,
                            'conversation_boundaries': self._find_conversation_boundaries(encoding.ids, full_text)
                        })

    def _find_conversation_boundaries(self, token_ids: List[int], text: str) -> List[Tuple[int, int]]:
        """Find natural chunk boundaries (e.g., speaker turns)"""
        boundaries = []
        # Simple heuristic - you could make this more sophisticated
        # Look for newlines or speaker markers
        decoded_tokens = [self.tokenizer.decode([tid]) for tid in token_ids]

        current_chunk_start = 0
        for i, token in enumerate(decoded_tokens):
            if '\n' in token or ':' in token:
                if i > current_chunk_start + 2:  # Minimum chunk size
                    boundaries.append((current_chunk_start, i))
                    current_chunk_start = i + 1

        # Add final chunk
        if current_chunk_start < len(token_ids) - 1:
            boundaries.append((current_chunk_start, len(token_ids) - 1))

        return boundaries

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = example['input_ids'][:]  # Copy to avoid modifying original
        boundaries = example.get('conversation_boundaries', [])[:]  # Copy

        # Truncate or pad
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            # Also truncate boundaries
            boundaries = [(s, e) for s, e in boundaries if s < self.max_length and e < self.max_length]
        else:
            # Pad input_ids
            padding_length = self.max_length - len(input_ids)
            input_ids = input_ids + [0] * padding_length

        # Convert to tensors
        input_tensor = torch.tensor(input_ids, dtype=torch.long)

        return {
            'input_ids': input_tensor,
            'labels': input_tensor.clone(),
            'boundaries': boundaries  # Keep as list of tuples
        }

class TemporalBindingTrainer:
    """Trainer that leverages temporal binding for better language modeling"""

    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset] = None,
            learning_rate: float = 5e-5,
            batch_size: int = 8,
            gradient_accumulation_steps: int = 4,
            max_grad_norm: float = 1.0,
            warmup_steps: int = 1000,
            eval_steps: int = 500,
            save_steps: int = 1000,
            num_epochs: int = 3,
            device: str = DEVICE,
            output_dir: str = "./checkpoints",
            use_wandb: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # In TemporalBindingTrainer.__init__, update the DataLoaders:
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate_fn  # Add this
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn  # Add this
        ) if val_dataset else None

        # Optimizer with different LR for binding components
        binding_params = []
        other_params = []

        for name, param in model.named_parameters():
            if 'bind' in name or 'pattern' in name or 'context_modulator' in name:
                binding_params.append(param)
            else:
                other_params.append(param)

        self.optimizer = torch.optim.AdamW([
            {'params': binding_params, 'lr': learning_rate * 2},  # Higher LR for binding
            {'params': other_params, 'lr': learning_rate}
        ], weight_decay=0.01)

        # Learning rate scheduler
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = self.get_linear_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Metrics
        self.train_losses = []
        self.val_losses = []
        self.binding_accuracies = []

        # Wandb
        if use_wandb:
            wandb.init(project="temporal-binding-lm", config={
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'model_dim': model.dim if hasattr(model, 'dim') else 'unknown'
            })

    def get_linear_schedule_with_warmup(self, optimizer, warmup_steps, total_steps):
        """Linear warmup then linear decay"""

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - step) / float(max(1, total_steps - warmup_steps)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def compute_binding_accuracy(self, model, batch):
        """Measure how well the model's binding aligns with natural boundaries"""
        model.eval()
        with torch.no_grad():
            input_ids = batch['input_ids'].to(self.device)
            boundaries = batch['boundaries']

            # Get embeddings
            tok_emb = model.token_emb(input_ids)
            pos_idx = torch.arange(input_ids.size(1), device=self.device).unsqueeze(0)
            pos_emb = model.pos_emb(pos_idx)
            x = tok_emb + pos_emb

            # Get binding from first scale
            binding_field = model.binder.binding_fields[0]
            if hasattr(binding_field, 'improved_field'):
                mask, strength, _ = binding_field.improved_field(x)
            else:
                mask, strength = binding_field(x)

            # Compare with ground truth boundaries
            accuracies = []
            for b in range(input_ids.size(0)):
                # Fix: boundaries is a list of boundary lists, not a tensor
                if b < len(boundaries) and boundaries[b]:
                    batch_boundaries = boundaries[b]

                    # Handle different boundary formats
                    if isinstance(batch_boundaries, torch.Tensor):
                        batch_boundaries = batch_boundaries.tolist()

                    # Check if model binds at expected boundaries
                    correct = 0
                    total = 0

                    for boundary in batch_boundaries:
                        # Handle both tuple and list formats
                        if isinstance(boundary, (list, tuple)) and len(boundary) >= 2:
                            start, end = boundary[0], boundary[1]
                        else:
                            continue  # Skip malformed boundaries

                        if start < mask.size(1) and end <= mask.size(1):
                            # Model should NOT bind at chunk boundaries
                            if mask[b, start] == 0:
                                correct += 1
                            total += 1

                            # Model SHOULD bind within chunks
                            for i in range(start + 1, min(end, mask.size(1))):
                                if mask[b, i] == 1:
                                    correct += 1
                                total += 1

                    if total > 0:
                        accuracies.append(correct / total)

            return np.mean(accuracies) if accuracies else 0.0

    def train_step(self, batch) -> Dict[str, float]:
        """Single training step with binding-aware loss"""
        self.model.train()

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Shift labels for autoregressive training
        shift_logits = self.model(input_ids[:, :-1])
        shift_labels = labels[:, 1:]

        # Standard LM loss
        lm_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=0  # Padding token
        )

        # Optional: Add binding regularization loss
        # This encourages the model to bind at semantically meaningful boundaries
        binding_loss = 0.0
        if hasattr(self.model, 'binder'):
            # Get binding strengths
            tok_emb = self.model.token_emb(input_ids[:, :-1])
            pos_idx = torch.arange(input_ids.size(1) - 1, device=self.device).unsqueeze(0)
            pos_emb = self.model.pos_emb(pos_idx)
            x = tok_emb + pos_emb

            binding_field = self.model.binder.binding_fields[0]
            if hasattr(binding_field, 'improved_field'):
                _, strength, analysis = binding_field.improved_field(x)

                # Regularize pattern matching to be sparse but strong
                if 'pattern_boost' in analysis:
                    pattern_sparsity = torch.mean(torch.abs(analysis['pattern_boost']))
                    binding_loss += 0.1 * pattern_sparsity

                # Encourage adaptive thresholds to be reasonable
                if 'adaptive_thresholds' in analysis:
                    threshold_var = torch.var(analysis['adaptive_thresholds'])
                    binding_loss += 0.05 * threshold_var

        # Total loss
        total_loss = lm_loss + binding_loss

        # Backward
        total_loss.backward()

        return {
            'loss': total_loss.item(),
            'lm_loss': lm_loss.item(),
            'binding_loss': binding_loss.item() if isinstance(binding_loss, torch.Tensor) else binding_loss
        }

    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        if not self.val_loader:
            return {}

        self.model.eval()
        total_loss = 0
        total_binding_acc = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Get loss
                shift_logits = self.model(input_ids[:, :-1])
                shift_labels = labels[:, 1:]

                loss = F.cross_entropy(
                    shift_logits.reshape(-1, shift_logits.size(-1)),
                    shift_labels.reshape(-1),
                    ignore_index=0
                )

                total_loss += loss.item()

                # Get binding accuracy
                binding_acc = self.compute_binding_accuracy(self.model, batch)
                total_binding_acc += binding_acc

                num_batches += 1

        return {
            'val_loss': total_loss / num_batches,
            'val_binding_accuracy': total_binding_acc / num_batches,
            'val_perplexity': np.exp(total_loss / num_batches)
        }

    def train(self, resume_from_checkpoint=True):
        """Main training loop with checkpoint resumption"""
        global_step = 0
        best_val_loss = float('inf')
        start_epoch = 0
        start_batch = 0

        # Try to resume from checkpoint
        if resume_from_checkpoint:
            resume_state = self.load_checkpoint()
            if resume_state:
                global_step = resume_state['step']
                start_epoch = resume_state['epoch']
                start_batch = resume_state['batch']

                # Load best val loss if available
                if hasattr(self, 'best_val_loss'):
                    best_val_loss = self.best_val_loss

        for epoch in range(start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Training
            epoch_losses = []
            self.model.train()

            progress_bar = tqdm(self.train_loader, desc="Training")
            for step, batch in enumerate(progress_bar):
                # Skip batches if resuming mid-epoch
                if epoch == start_epoch and step < start_batch:
                    continue

                # Forward & backward
                metrics = self.train_step(batch)
                epoch_losses.append(metrics['loss'])

                # Gradient accumulation
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    global_step += 1

                    # Update progress bar
                    progress_bar.set_postfix({
                        'loss': np.mean(epoch_losses[-100:]),
                        'lr': self.scheduler.get_last_lr()[0]
                    })

                    # Log to wandb
                    if wandb.run:
                        wandb.log({
                            'train/loss': metrics['loss'],
                            'train/lm_loss': metrics['lm_loss'],
                            'train/binding_loss': metrics['binding_loss'],
                            'train/lr': self.scheduler.get_last_lr()[0]
                        }, step=global_step)

                    # Evaluate
                    if global_step % self.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        print(f"\nStep {global_step}: {eval_metrics}")

                        if wandb.run:
                            wandb.log({
                                'eval/loss': eval_metrics.get('val_loss', 0),
                                'eval/binding_accuracy': eval_metrics.get('val_binding_accuracy', 0),
                                'eval/perplexity': eval_metrics.get('val_perplexity', 0)
                            }, step=global_step)

                        # Save best model
                        if eval_metrics.get('val_loss', float('inf')) < best_val_loss:
                            best_val_loss = eval_metrics['val_loss']
                            self.best_val_loss = best_val_loss  # Store for next resume
                            self.save_checkpoint(global_step, is_best=True)

                    # Regular checkpoint
                    if global_step % self.save_steps == 0:
                        self.save_checkpoint(global_step)

            # Reset start_batch for subsequent epochs
            start_batch = 0

            # End of epoch evaluation
            print(f"\nEpoch {epoch + 1} complete. Average loss: {np.mean(epoch_losses):.4f}")

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint with all necessary state"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': {
                'vocab_size': self.model.token_emb.num_embeddings,
                'dim': self.model.token_emb.embedding_dim,
                'num_layers': len(self.model.transformer_layers),
                'num_heads': self.model.transformer_layers[0].attention.num_heads,
                'num_scales': self.model.num_scales,
                'max_seq_len': self.model.pos_emb.num_embeddings,
            },
            'best_val_loss': getattr(self, 'best_val_loss', float('inf'))
        }

        if is_best:
            path = self.output_dir / 'best_model.pt'
        else:
            path = self.output_dir / f'checkpoint_step_{step}.pt'

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, checkpoint_path=None):
        """Load checkpoint and resume training"""
        # If no path specified, find the latest checkpoint
        if checkpoint_path is None:
            # Look for best model first
            best_path = self.output_dir / 'best_model.pt'
            if best_path.exists():
                checkpoint_path = best_path
            else:
                # Find latest numbered checkpoint
                checkpoints = list(self.output_dir.glob('checkpoint_step_*.pt'))
                if checkpoints:
                    # Sort by step number
                    checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
                    checkpoint_path = checkpoints[-1]
                else:
                    print("No checkpoint found to resume from")
                    return None

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Load scheduler state
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # Get training state
        resume_step = checkpoint.get('step', 0)

        # Calculate which epoch and batch to resume from
        steps_per_epoch = len(self.train_loader) // self.gradient_accumulation_steps
        resume_epoch = resume_step // steps_per_epoch
        resume_batch = (resume_step % steps_per_epoch) * self.gradient_accumulation_steps

        print(f"Resuming from step {resume_step} (epoch {resume_epoch}, batch {resume_batch})")

        return {
            'step': resume_step,
            'epoch': resume_epoch,
            'batch': resume_batch
        }

    def log_binding_patterns(self, model, batch, step):
        """Log binding patterns to wandb for visualization"""
        with torch.no_grad():
            # Get a sample
            sample_idx = 0
            tokens = batch['input_ids'][sample_idx]

            # Get binding mask and strength
            tok_emb = model.token_emb(tokens.unsqueeze(0))
            pos_emb = model.pos_emb(torch.arange(len(tokens), device=tokens.device).unsqueeze(0))
            x = tok_emb + pos_emb

            binding_field = model.binder.binding_fields[0]
            if hasattr(binding_field, 'improved_field'):
                mask, strength, analysis = binding_field.improved_field(x)
            else:
                mask, strength = binding_field(x)

            # Decode tokens for visualization
            token_strs = [self.tokenizer.decode([t.item()]) for t in tokens[:50]]  # First 50

            # Create binding visualization
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

            # Token sequence with binding
            ax1.imshow(mask[0, :50].cpu().numpy().reshape(1, -1),
                       aspect='auto', cmap='RdGn')
            ax1.set_xticks(range(len(token_strs)))
            ax1.set_xticklabels(token_strs, rotation=45, ha='right')
            ax1.set_title('Binding Mask (Green = Bound)')

            # Binding strength
            if strength.size(1) > 0:
                ax2.plot(strength[0, :49].cpu().numpy())
                ax2.axhline(y=0.45, color='r', linestyle='--', label='Threshold')
                ax2.set_title('Binding Strength')
                ax2.legend()

            wandb.log({"binding_patterns": wandb.Image(fig)}, step=step)

    # Add this diagnostic function to your trainer
    def diagnose_binding(self, num_samples=5):
        """Check if binding is actually happening"""
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_samples:
                    break

                input_ids = batch['input_ids'].to(self.device)

                # Get binding masks
                tok_emb = self.model.token_emb(input_ids)
                pos_emb = self.model.pos_emb(torch.arange(input_ids.size(1), device=self.device).unsqueeze(0))
                x = tok_emb + pos_emb

                binding_field = self.model.binder.binding_fields[0]
                if hasattr(binding_field, 'improved_field'):
                    mask, strength, _ = binding_field.improved_field(x)
                else:
                    mask, strength = binding_field(x)

                # Print statistics
                print(f"\nBatch {i}:")
                print(f"  Binding mask mean: {mask.mean().item():.4f}")
                print(f"  Binding strength mean: {strength.mean().item():.4f}")
                print(f"  Binding strength std: {strength.std().item():.4f}")
                print(f"  % tokens bound: {(mask > 0).float().mean().item() * 100:.1f}%")

                # Show sample
                sample_mask = mask[0, :20].cpu().numpy()
                sample_strength = strength[0, :19].cpu().numpy()
                print(f"  Sample mask: {sample_mask}")
                print(f"  Sample strength: {sample_strength}")



# Main training script
def main():
    # Load tokenizer
    tokenizer = Tokenizer.from_file("./personachat_bpe_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    # Create model with improved binding
    model = GenerativeTemporalHierarchicalLM(
        vocab_size=vocab_size,
        dim=512,
        num_layers=8,
        num_heads=8,
        num_scales=3,
        max_seq_len=512,
        use_improved_binder=True
    ).to(DEVICE)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    # Create datasets
    train_dataset = ConversationDataset(
        tokenizer_path="./personachat_bpe_tokenizer.json",
        max_length=512,
        split="train"
    )

    val_dataset = ConversationDataset(
        tokenizer_path="./personachat_bpe_tokenizer.json",
        max_length=512,
        split="validation"
    )

    print(f"Train examples: {len(train_dataset)}")
    print(f"Val examples: {len(val_dataset)}")

    # Create trainer
    trainer = TemporalBindingTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=5e-5,
        batch_size=8,
        gradient_accumulation_steps=4,
        num_epochs=3,
        eval_steps=500,
        save_steps=1000,
        use_wandb=True
    )

    resume_from_checkpoint = True
    # Load checkpoint if resuming
    if resume_from_checkpoint:
        trainer.load_checkpoint()

    # Apply improvements
    patch_binding_loss(trainer)
    add_binding_diagnostics(trainer)

    # Adjust thresholds on the model
    adjust_binding_threshold(trainer.model)

    # Run diagnostics to see current state
    trainer.diagnose_binding()

    # Continue training with improvements
    trainer.train(resume_from_checkpoint=False)

    # Test generation after training
    print("\n" + "=" * 50)
    print("Testing generation with trained model...")

    test_prompt = "Person1: Hi! How are you doing today?\nPerson2:"
    encoding = tokenizer.encode(test_prompt)
    input_ids = torch.tensor([encoding.ids], device=DEVICE)

    generated, binding_info = model.generate(
        input_ids,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.95,
        return_binding_info=True
    )

    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Generated: {generated_text}")

    # Analyze binding patterns
    if binding_info:
        print("\nBinding analysis of generated text:")
        for i, info in enumerate(binding_info[:5]):  # First 5 tokens
            if 'scale_binding' in info and info['scale_binding']:
                scale_0 = info['scale_binding'][0]
                if 'strength' in scale_0:
                    print(f"Token {i}: binding strength = {scale_0['strength'][0, -1].item():.3f}")


if __name__ == "__main__":
    main()