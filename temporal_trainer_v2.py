import os
from collections import defaultdict

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
import random

# Import your existing models and components
from temporal_binding_demo import DEVICE, IGNORE_INDEX
from thbt_averaged_signals import build_temporal_model
from temporal_trainer import (
    ConversationDataset,
    custom_collate_fn,
    GenerativeTemporalHierarchicalLM
)


class ImprovedTemporalTrainer:
    """Trainer that prevents cheating through various techniques"""

    def __init__(
            self,
            model: nn.Module,
            train_dataset: Dataset,
            val_dataset: Optional[Dataset] = None,
            tokenizer=None,  # ADD THIS LINE
            learning_rate: float = 5e-5,
            batch_size: int = 8,
            gradient_accumulation_steps: int = 4,
            max_grad_norm: float = 1.0,
            warmup_steps: int = 1000,
            eval_steps: int = 500,
            save_steps: int = 1000,
            num_epochs: int = 3,
            device: str = DEVICE,
            output_dir: str = "./checkpoints_v2",
            use_wandb: bool = True,
            scheduled_sampling_prob: float = 0.0,  # Will increase during training
    ):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = tokenizer  # ADD THIS LINE
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Current epoch for scheduling
        self.current_epoch = 0
        self.global_step = 0

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=custom_collate_fn
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=custom_collate_fn
        ) if val_dataset else None

        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Scheduler
        total_steps = len(self.train_loader) * num_epochs // gradient_accumulation_steps
        self.scheduler = self.get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps
        )

        # Initialize wandb
        if use_wandb:
            wandb.init(project="temporal-binding-lm-v2", config={
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'scheduled_sampling': True,
                'adversarial_training': True,
            })

    def get_cosine_schedule_with_warmup(self, optimizer, warmup_steps, total_steps):
        """Cosine learning rate schedule with warmup"""

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def scheduled_sampling_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training with scheduled sampling using stochastic predictions"""

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)
        B, T = input_ids.shape

        sampling_prob = min(0.5, 0.1 * (self.current_epoch / self.num_epochs))

        # Standard forward pass
        outputs = self.model(input_ids[:, :-1])
        teacher_loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=0
        )

        if sampling_prob > 0 and T > 10:  # Only for longer sequences
            with torch.no_grad():
                # MODIFIED: Use sampling instead of argmax
                temperature = 0.8
                logits = outputs / temperature

                # Apply top-k filtering
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)

                # Sample from top-k
                probs = F.softmax(top_k_logits, dim=-1)
                sampled_indices = torch.multinomial(probs.view(-1, top_k), 1).view(B, -1)
                predictions = torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)

                # Create mixed input with sampled predictions
                replace_mask = torch.rand(B, outputs.size(1), device=self.device) < sampling_prob
                replace_mask[:, :5] = False  # Keep first tokens

                mixed_input = input_ids.clone()
                mixed_input[:, 1:][replace_mask] = predictions[replace_mask]

            # Forward with mixed input
            mixed_outputs = self.model(mixed_input[:, :-1])
            mixed_loss = F.cross_entropy(
                mixed_outputs.reshape(-1, mixed_outputs.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=0
            )

            total_loss = (1 - sampling_prob * 0.3) * teacher_loss + (sampling_prob * 0.3) * mixed_loss
        else:
            total_loss = teacher_loss

        total_loss.backward()

        return {
            'loss': total_loss.item(),
            'sampling_prob': sampling_prob
        }

    def binding_regularization_loss(self, model, strength=0.1):
        """Regularize binding patterns to prevent over-chunking"""
        reg_loss = 0.0

        # Regularize pattern memory to prevent memorization
        if hasattr(model, 'binder'):
            for field in model.binder.binding_fields:
                if hasattr(field, 'improved_field'):
                    # L2 regularization on pattern memory
                    pattern_memory = field.improved_field.pattern_memory
                    reg_loss += strength * torch.norm(pattern_memory, p=2)

                    # Encourage diversity in patterns
                    pattern_similarity = torch.matmul(
                        F.normalize(pattern_memory, dim=-1),
                        F.normalize(pattern_memory, dim=-1).T
                    )
                    # Penalize high similarity between different patterns
                    off_diagonal = pattern_similarity - torch.eye(
                        pattern_similarity.size(0),
                        device=pattern_similarity.device
                    )
                    reg_loss += strength * torch.mean(torch.abs(off_diagonal))

        return reg_loss

    def adversarial_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Training with input perturbations to improve robustness"""

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # 1. Normal forward pass
        outputs_normal = self.model(input_ids[:, :-1])
        loss_normal = F.cross_entropy(
            outputs_normal.reshape(-1, outputs_normal.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=0
        )

        # 2. Create adversarial inputs by random token substitution
        # Replace 10% of tokens with random tokens
        mask = torch.rand_like(input_ids.float()) < 0.1
        # Avoid replacing special tokens (0-10)
        mask = mask & (input_ids > 10)

        random_tokens = torch.randint(
            10, 1000, size=input_ids.shape, device=self.device
        )
        adversarial_ids = torch.where(mask, random_tokens, input_ids)

        # 3. Forward pass with adversarial inputs
        outputs_adv = self.model(adversarial_ids[:, :-1])
        loss_adv = F.cross_entropy(
            outputs_adv.reshape(-1, outputs_adv.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=0
        )

        # 4. Combined loss (normal + adversarial)
        total_loss = loss_normal + 0.3 * loss_adv

        # Backward
        total_loss.backward()

        return {
            'loss': total_loss.item(),
            'loss_normal': loss_normal.item(),
            'loss_adversarial': loss_adv.item()
        }

    def generation_aware_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate model on its ability to generate, not just predict next token"""

        input_ids = batch['input_ids'].to(self.device)

        # Use first 30% as prompt, rest as target
        prompt_len = max(5, int(input_ids.size(1) * 0.3))
        prompts = input_ids[:, :prompt_len]
        targets = input_ids[:, prompt_len:]

        # Generate autoregressively
        current = prompts
        total_loss = 0
        num_correct = 0
        num_repetitions = 0

        for t in range(min(targets.size(1), 20)):  # Limit generation length
            outputs = self.model(current)
            logits = outputs[:, -1, :]

            # Loss against target
            if t < targets.size(1):
                target_t = targets[:, t]
                loss_t = F.cross_entropy(logits, target_t, ignore_index=0)
                total_loss += loss_t

                # Track accuracy
                predictions = torch.argmax(logits, dim=-1)
                num_correct += (predictions == target_t).float().mean()

            # For next step, use model's own prediction
            with torch.no_grad():
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

                # Track repetitions
                if current.size(1) > 0:
                    num_repetitions += (next_token.squeeze() == current[:, -1]).float().mean()

                current = torch.cat([current, next_token], dim=1)

        # Backward
        if targets.size(1) > 0:
            avg_loss = total_loss / min(targets.size(1), 20)
            avg_loss.backward()

        return {
            'loss': avg_loss.item() if targets.size(1) > 0 else 0,
            'generation_accuracy': (num_correct / min(targets.size(1), 20)).item(),
            'repetition_rate': (num_repetitions / min(targets.size(1), 20)).item()
        }

    def train_step(self, batch: Dict[str, torch.Tensor], step: int) -> Dict[str, float]:
        """Choose training strategy based on step"""

        # For the first epoch, just use standard training
        if self.current_epoch == 0:
            return self.standard_training_step(batch)

        # Rotate through different training strategies
        strategy = step % 4

        if strategy == 0:
            # Standard training
            return self.standard_training_step(batch)
        elif strategy == 1:
            # Scheduled sampling (if epoch > 0)
            return self.scheduled_sampling_step(batch)
        elif strategy == 2:
            # Adversarial training
            return self.adversarial_step(batch)
        else:
            # Generation-aware training
            return self.generation_aware_step(batch)

    def standard_training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Enhanced training step with all improvements"""

        input_ids = batch['input_ids'].to(self.device)
        labels = batch['labels'].to(self.device)

        # Forward pass
        outputs = self.model(input_ids[:, :-1])

        # Multi-scale repetition-aware loss
        loss, ce_loss, rep_penalty = self.repetition_aware_loss(
            outputs,
            labels[:, 1:],
            input_ids[:, :-1]
        )

        # Add binding regularization
        binding_reg = self.binding_regularization_loss(self.model, strength=0.01)

        # Total loss
        total_loss = loss + binding_reg

        # Backward
        total_loss.backward()

        return {
            'loss': total_loss.item(),
            'ce_loss': ce_loss,
            'rep_penalty': rep_penalty,
            'binding_reg': binding_reg.item() if isinstance(binding_reg, torch.Tensor) else binding_reg,
            'skipped': 0.0
        }

    def evaluate_generation_quality(self, num_samples: int = 10) -> Dict[str, float]:
        """Evaluate actual generation quality, not just perplexity"""

        self.model.eval()

        test_prompts = [
            "I love",
            "The weather is",
            "My favorite",
            "I work as a",
            "Hello! How are you",
        ]

        all_repetition_rates = []
        all_unique_tokens = []

        with torch.no_grad():
            for prompt in test_prompts:
                # Encode prompt
                encoding = self.tokenizer.encode(prompt)
                input_ids = torch.tensor([encoding.ids], device=self.device)

                # Generate
                generated = []
                current = input_ids

                for _ in range(30):
                    outputs = self.model(current)
                    next_token = torch.argmax(outputs[:, -1, :], dim=-1)
                    generated.append(next_token.item())

                    current = torch.cat([current, next_token.unsqueeze(0)], dim=1)

                    # Break if EOS
                    if next_token.item() == 3:  # EOS token
                        break

                # Calculate metrics
                if len(generated) > 1:
                    # Repetition rate
                    repetitions = sum(1 for i in range(1, len(generated))
                                      if generated[i] == generated[i - 1])
                    rep_rate = repetitions / len(generated)
                    all_repetition_rates.append(rep_rate)

                    # Unique token ratio
                    unique_ratio = len(set(generated)) / len(generated)
                    all_unique_tokens.append(unique_ratio)

        self.model.train()

        return {
            'avg_repetition_rate': np.mean(all_repetition_rates) if all_repetition_rates else 1.0,
            'avg_unique_token_ratio': np.mean(all_unique_tokens) if all_unique_tokens else 0.0,
        }

    def train(self):
        """Main training loop with anti-cheating measures"""

        best_combined_score = float('inf')

        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print(f"Scheduled sampling prob: {min(0.5, 0.1 * (epoch / self.num_epochs)):.3f}")

            # Training
            self.model.train()
            epoch_metrics = defaultdict(list)

            progress_bar = tqdm(self.train_loader, desc="Training")
            for step, batch in enumerate(progress_bar):

                # Accumulate gradients
                metrics = self.train_step(batch, step)

                for key, value in metrics.items():
                    epoch_metrics[key].append(value)

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Check for NaN gradients
                    total_norm = 0
                    for p in self.model.parameters():
                        if p.grad is not None:
                            param_norm = p.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2

                            # Check for NaN
                            if torch.isnan(p.grad).any():
                                print("WARNING: NaN gradients detected! Skipping update.")
                                self.optimizer.zero_grad()
                                continue

                    total_norm = total_norm ** 0.5

                    if total_norm > 100:
                        print(f"WARNING: Large gradient norm: {total_norm:.2f}")

                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    self.global_step += 1

                    # Update progress bar
                    avg_loss = np.mean(epoch_metrics['loss'][-100:])
                    progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})

                    # Log to wandb
                    if wandb.run and self.global_step % 10 == 0:
                        log_dict = {
                            f'train/{k}': np.mean(v[-100:])
                            for k, v in epoch_metrics.items()
                        }
                        log_dict['train/lr'] = self.scheduler.get_last_lr()[0]
                        wandb.log(log_dict, step=self.global_step)

                    # Evaluate
                    if self.global_step % self.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        self.show_generation_examples()

                        # Evaluate generation quality
                        gen_metrics = self.evaluate_generation_quality()
                        eval_metrics.update(gen_metrics)

                        # Combined score (lower is better)
                        combined_score = (
                                eval_metrics['val_loss'] * 0.2 +
                                eval_metrics['avg_repetition_rate'] * 5.0 +
                                (1 - eval_metrics['avg_unique_token_ratio']) * 3.0
                        )

                        print(f"\nStep {self.global_step}:")
                        print(f"  Val Loss: {eval_metrics['val_loss']:.4f}")
                        print(f"  Val PPL: {eval_metrics['val_perplexity']:.4f}")
                        print(f"  Repetition Rate: {eval_metrics['avg_repetition_rate']:.3f}")
                        print(f"  Unique Token Ratio: {eval_metrics['avg_unique_token_ratio']:.3f}")
                        print(f"  Combined Score: {combined_score:.4f}")

                        if wandb.run:
                            wandb.log({
                                'eval/loss': eval_metrics['val_loss'],
                                'eval/perplexity': eval_metrics['val_perplexity'],
                                'eval/repetition_rate': eval_metrics['avg_repetition_rate'],
                                'eval/unique_token_ratio': eval_metrics['avg_unique_token_ratio'],
                                'eval/combined_score': combined_score,
                            }, step=self.global_step)

                        # Save best model based on combined score
                        if combined_score < best_combined_score:
                            best_combined_score = combined_score
                            self.save_checkpoint(self.global_step, is_best=True)
                            print("  → Saved new best model!")

                    # Regular checkpoint
                    if self.global_step % self.save_steps == 0:
                        self.save_checkpoint(self.global_step)

            # End of epoch
            print(f"\nEpoch {epoch + 1} complete.")
            print(f"Average metrics:")
            for key, values in epoch_metrics.items():
                print(f"  {key}: {np.mean(values):.4f}")

    def evaluate(self) -> Dict[str, float]:
        """Standard evaluation"""
        if not self.val_loader:
            return {}

        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids[:, :-1])
                loss = F.cross_entropy(
                    outputs.reshape(-1, outputs.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=0
                )

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches

        return {
            'val_loss': avg_loss,
            'val_perplexity': np.exp(avg_loss)
        }

    def show_generation_examples(self):
        """Show actual generated text during evaluation"""
        self.model.eval()

        test_prompts = ["I love", "The weather", "My favorite"]

        print("\nGeneration examples:")
        for prompt in test_prompts:
            encoding = self.tokenizer.encode(prompt)
            input_ids = torch.tensor([encoding.ids], device=self.device)

            generated = []
            current = input_ids

            for _ in range(20):
                with torch.no_grad():
                    outputs = self.model(current)
                    # Add temperature and top-k sampling
                    logits = outputs[:, -1, :] / 0.8
                    top_k = torch.topk(logits, 50).indices
                    probs = F.softmax(logits.scatter(-1, top_k, float('-inf')), dim=-1)
                    next_token = torch.multinomial(probs, 1)

                    generated.append(next_token.item())
                    current = torch.cat([current, next_token], dim=1)

            text = self.tokenizer.decode(input_ids[0].tolist() + generated)
            print(f"  '{prompt}' -> '{text}'")

    def repetition_aware_loss(self, outputs, labels, input_ids):
        """Multi-scale repetition penalty"""

        # Standard CE loss
        ce_loss = F.cross_entropy(
            outputs.reshape(-1, outputs.size(-1)),
            labels.reshape(-1),
            ignore_index=0
        )

        # Get logits for repetition penalty
        B, T, V = outputs.shape

        # Penalize high probability on repeated tokens
        repetition_loss = 0.0

        # 1. Immediate repetition penalty on logits
        if T > 1:
            # For each position, penalize if previous token has high probability
            prev_tokens = input_ids[:, :T]  # Previous tokens

            for t in range(1, T):
                # Get logits at position t
                logits_t = outputs[:, t, :]

                # Previous token
                prev_token = prev_tokens[:, t]

                # Probability assigned to previous token
                prev_token_probs = F.softmax(logits_t, dim=-1)
                repeat_probs = torch.gather(prev_token_probs, 1, prev_token.unsqueeze(1)).squeeze()

                # Add to loss if probability is high
                repetition_loss += torch.mean(torch.relu(repeat_probs - 0.1))  # Penalize if > 10%

        # 2. N-gram repetition penalty
        if T > 3:
            # Check for repeated bigrams
            for t in range(2, T):
                if t >= 2:
                    # Current bigram prediction
                    curr_logits = outputs[:, t - 1:t + 1, :]
                    curr_pred = torch.argmax(curr_logits, dim=-1)

                    # Check against previous bigrams
                    for prev_t in range(max(0, t - 10), t - 2):
                        prev_bigram = input_ids[:, prev_t:prev_t + 2]

                        # Penalize if predicting same bigram
                        bigram_match = (curr_pred == prev_bigram).all(dim=1).float()
                        repetition_loss += 0.5 * bigram_match.mean()

        # Total loss with repetition penalty
        total_loss = ce_loss + 3.0 * repetition_loss / max(T, 1)

        return total_loss, ce_loss.item(), repetition_loss.item()

    def save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'epoch': self.current_epoch,
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
            'training_config': {
                'scheduled_sampling': True,
                'adversarial_training': True,
                'generation_aware': True,
            }
        }

        if is_best:
            path = self.output_dir / 'best_model_v2.pt'
        else:
            path = self.output_dir / f'checkpoint_step_{step}.pt'

        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")


def main():
    """Main training function with improved training scheme"""

    # Load tokenizer
    tokenizer = Tokenizer.from_file("./personachat_bpe_tokenizer.json")
    vocab_size = tokenizer.get_vocab_size()

    # Create model (you can also load a pretrained one and continue training)
    model_config = {
        'vocab_size': vocab_size,
        'dim': 512,
        'num_layers': 8,
        'num_heads': 8,
        'num_scales': 3,
        'max_seq_len': 512,
    }

    model = build_temporal_model(model_config)

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

    if os.path.exists("checkpoints_v2/checkpoint_step_500.pt"):
        print("Loading checkpoint from step 500 (better generation)...")
        checkpoint = torch.load("checkpoints_v2/checkpoint_step_500.pt")
        model = build_temporal_model(checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        # Create improved trainer with tokenizer
        trainer = ImprovedTemporalTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            learning_rate=3e-5,
            batch_size=4,  # Reduced from 8
            gradient_accumulation_steps=8,  # Increased to compensate
            num_epochs=5,
            eval_steps=500,
            save_steps=1000,
            use_wandb=True
        )
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.global_step = checkpoint['step']
        trainer.current_epoch = checkpoint['epoch']
    else:
        trainer = ImprovedTemporalTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            tokenizer=tokenizer,
            learning_rate=3e-5,
            batch_size=4,  # Reduced from 8
            gradient_accumulation_steps=8,  # Increased to compensate
            num_epochs=5,
            eval_steps=500,
            save_steps=1000,
            use_wandb=True
        )
    # Train!
    print("\nStarting improved training...")
    trainer.train()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
