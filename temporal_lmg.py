import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict
import math

# Import all your sophisticated components
from temporal_binding_demo import (
    TemporalHierarchicalTransformer,
    HierarchicalBinder,
    TemporalScale,
    TransformerLayer,
    BIND_THRESHOLD
)
from thbt_averaged_signals import (
    ImprovedBindingField,
    ImprovedHierarchicalBinder,
    BindingFieldWrapper
)


class GenerativeTemporalHierarchicalLM(TemporalHierarchicalTransformer):
    """
    Extends your TemporalHierarchicalTransformer for generation,
    utilizing ALL the temporal binding machinery including:
    - Multi-scale hierarchical processing
    - Improved binding fields with context modulation
    - Bidirectional influence between scales
    - Attention-based chunk aggregation
    """

    def __init__(
            self,
            vocab_size: int,
            dim: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            num_scales: int = 3,
            max_seq_len: int = 2048,
            update_frequencies: Optional[List[int]] = None,
            fixed_bind_threshold: float = BIND_THRESHOLD,
            use_improved_binder: bool = True
    ):
        # Initialize parent class
        super().__init__(
            vocab_size=vocab_size,
            dim=dim,
            num_layers=num_layers,
            num_heads=num_heads,
            num_scales=num_scales,
            max_seq_len=max_seq_len,
            update_frequencies=update_frequencies,
            fixed_bind_threshold=fixed_bind_threshold
        )

        # Replace with improved binder if requested
        if use_improved_binder:
            self._upgrade_to_improved_binder()

        # Add generation-specific components
        self.cache_manager = MultiScaleCacheManager(num_scales, num_layers)

    def _upgrade_to_improved_binder(self):
        """Replace standard binding fields with improved versions"""
        for i in range(len(self.binder.binding_fields)):
            old_field = self.binder.binding_fields[i]
            improved_field = ImprovedBindingField(
                dim=self.binder.dim,
                max_chunk_size=old_field.max_chunk_size,
                base_threshold=old_field.bind_threshold.item()
            )
            # Copy learned parameters if they exist
            if hasattr(old_field, 'bind_key'):
                with torch.no_grad():
                    improved_field.bind_key.weight.copy_(old_field.bind_key.weight)
                    improved_field.bind_query.weight.copy_(old_field.bind_query.weight)

            self.binder.binding_fields[i] = BindingFieldWrapper(improved_field)

    def forward_with_cache(
            self,
            x_indices: torch.Tensor,
            past_cache: Optional[Dict] = None,
            use_cache: bool = True,
            return_binding_info: bool = False
    ) -> Tuple[torch.Tensor, Dict, Optional[Dict]]:
        """
        Forward pass with caching for generation.
        Properly handles multi-scale representations and binding.
        """
        B, T = x_indices.shape
        device = x_indices.device

        # Get embeddings
        tok_emb = self.token_emb(x_indices)
        if past_cache and 'seen_tokens' in past_cache:
            # For cached generation, adjust position embeddings
            start_pos = past_cache['seen_tokens']
            pos_indices = torch.arange(start_pos, start_pos + T, device=device).unsqueeze(0)
        else:
            pos_indices = torch.arange(T, device=device).unsqueeze(0).repeat(B, 1)

        pos_emb = self.pos_emb(pos_indices)
        current_repr = tok_emb + pos_emb

        # Multi-scale processing with binding
        scale_representations = []
        scale_binding_info = []
        all_chunk_maps = []
        current_padding_mask = (
                    x_indices == self.token_emb.padding_idx) if self.token_emb.padding_idx is not None else None

        # Process through temporal scales
        for scale_idx in range(self.num_scales):
            # Retrieve cached representation if available
            if past_cache and f'scale_{scale_idx}_repr' in past_cache:
                # Concatenate with cached
                cached_repr = past_cache[f'scale_{scale_idx}_repr']
                current_repr = torch.cat([cached_repr, current_repr], dim=1)

            # Process through scale
            processed_repr = self.scales[scale_idx](current_repr, mask=current_padding_mask)
            scale_representations.append(processed_repr)

            # Apply binding and aggregation for next scale
            if scale_idx < self.num_scales - 1:
                # Get binding mask and strength with all the improvements
                binding_field = self.binder.binding_fields[scale_idx]
                if hasattr(binding_field, 'improved_field'):
                    binding_mask, binding_strength, binding_analysis = binding_field.improved_field(processed_repr)
                    scale_binding_info.append({
                        'mask': binding_mask,
                        'strength': binding_strength,
                        'analysis': binding_analysis
                    })
                else:
                    binding_mask, binding_strength = binding_field(processed_repr)
                    scale_binding_info.append({
                        'mask': binding_mask,
                        'strength': binding_strength
                    })

                # Aggregate chunks with attention-based method
                if hasattr(self.binder, 'aggregate_chunks'):
                    result = self.binder.aggregate_chunks(processed_repr, binding_mask, scale_idx)
                    if len(result) == 3:
                        aggregated_chunks, higher_scale_padding_mask, chunk_map = result
                        all_chunk_maps.append(chunk_map)
                    else:
                        aggregated_chunks, higher_scale_padding_mask = result
                        all_chunk_maps.append(None)

                    current_repr = aggregated_chunks
                    current_padding_mask = higher_scale_padding_mask

                if current_repr.size(1) == 0:
                    break

        # Apply bidirectional influence between scales
        influenced_representations = list(scale_representations)
        for scale_idx_lower in range(self.num_scales - 2, -1, -1):
            higher_scale_level = scale_idx_lower + 1
            if higher_scale_level < len(influenced_representations):
                higher_repr = influenced_representations[higher_scale_level]
                lower_repr = influenced_representations[scale_idx_lower]

                if higher_repr.size(1) > 0 and lower_repr.size(1) > 0:
                    chunk_map = all_chunk_maps[scale_idx_lower] if scale_idx_lower < len(all_chunk_maps) else None
                    influenced_lower = self.binder.apply_downward_influence(
                        higher_repr, lower_repr, scale_idx_lower, chunk_map
                    )
                    influenced_representations[scale_idx_lower] = influenced_lower

        # Use the influenced bottom scale for final processing
        final_repr = influenced_representations[0]

        # Process through transformer layers with caching
        if past_cache and 'layer_caches' in past_cache:
            layer_caches = past_cache['layer_caches']
        else:
            layer_caches = [None] * len(self.transformer_layers)

        new_layer_caches = []
        for i, layer in enumerate(self.transformer_layers):
            if isinstance(layer, TransformerLayerWithCache):
                final_repr, new_cache = layer(final_repr, past_kv=layer_caches[i], use_cache=use_cache)
                new_layer_caches.append(new_cache)
            else:
                # Fallback for original transformer layers
                final_repr = layer(final_repr)
                new_layer_caches.append(None)

        # Final output
        output = self.ln_f(final_repr)
        logits = self.lm_head(output)

        # Prepare cache for next step
        new_cache = None
        if use_cache:
            new_cache = {
                'seen_tokens': (past_cache['seen_tokens'] if past_cache else 0) + T,
                'layer_caches': new_layer_caches,
            }
            # Cache scale representations (only the most recent part)
            for scale_idx, scale_repr in enumerate(scale_representations):
                if scale_repr.size(1) > 0:
                    # Keep only last max_cache_len tokens
                    max_cache_len = 1024  # Configurable
                    if scale_repr.size(1) > max_cache_len:
                        new_cache[f'scale_{scale_idx}_repr'] = scale_repr[:, -max_cache_len:]
                    else:
                        new_cache[f'scale_{scale_idx}_repr'] = scale_repr

        # Prepare binding info if requested
        binding_info = None
        if return_binding_info:
            binding_info = {
                'scale_binding': scale_binding_info,
                'chunk_maps': all_chunk_maps,
                'influenced_representations': influenced_representations
            }

        return logits, new_cache, binding_info

    @torch.no_grad()
    def generate(
            self,
            input_ids: torch.Tensor,
            max_new_tokens: int = 100,
            temperature: float = 0.8,
            top_k: int = 50,
            top_p: float = 0.95,
            repetition_penalty: float = 1.1,
            return_binding_info: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[Dict]]]:
        """
        Generation that fully utilizes temporal binding.
        Returns generated tokens and optionally binding information for analysis.
        """
        self.eval()
        device = input_ids.device
        B = input_ids.shape[0]

        generated = input_ids
        past_cache = None
        all_binding_info = [] if return_binding_info else None

        for step in range(max_new_tokens):
            # Use only the new token(s) after first iteration
            if past_cache is not None:
                input_slice = generated[:, -1:]
            else:
                input_slice = generated

            # Forward with temporal binding
            logits, past_cache, binding_info = self.forward_with_cache(
                input_slice,
                past_cache=past_cache,
                use_cache=True,
                return_binding_info=return_binding_info
            )

            if return_binding_info and binding_info:
                all_binding_info.append(binding_info)

            # Get next token logits
            next_logits = logits[:, -1, :] / temperature

            # Apply repetition penalty
            for b in range(B):
                for token_id in set(generated[b].tolist()):
                    next_logits[b, token_id] /= repetition_penalty

            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                next_logits[indices_to_remove] = float('-inf')

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for EOS token
            if self.token_emb.padding_idx is not None:
                if (next_token == self.token_emb.padding_idx).all():
                    break

        return generated, all_binding_info


class TransformerLayerWithCache(TransformerLayer):
    """Extended transformer layer that supports KV caching"""

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = True
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        # Prepare keys and values
        if past_kv is not None:
            past_key, past_value = past_kv
            # Compute new K,V only for the new positions
            new_key = x  # In practice, apply key projection here
            new_value = x  # In practice, apply value projection here

            # Concatenate with past
            key = torch.cat([past_key, new_key], dim=1)
            value = torch.cat([past_value, new_value], dim=1)
        else:
            key = value = x

        # Standard transformer layer forward
        attn_out, _ = self.attention(x, key, value, key_padding_mask=mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)

        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)

        # Prepare cache
        new_cache = (key, value) if use_cache else None

        return x, new_cache


class MultiScaleCacheManager:
    """Manages caching across multiple temporal scales"""

    def __init__(self, num_scales: int, num_layers: int):
        self.num_scales = num_scales
        self.num_layers = num_layers
        self.reset()

    def reset(self):
        self.scale_caches = [{} for _ in range(self.num_scales)]
        self.layer_caches = [None for _ in range(self.num_layers)]
        self.seen_tokens = 0