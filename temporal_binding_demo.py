import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
import time
from collections import defaultdict

# --- Configuration & Constants ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IGNORE_INDEX = -100  # Standard ignore_index for CrossEntropyLoss
BIND_THRESHOLD = 0.35

class BindingField(nn.Module):
    """
    Computes binding potentials between adjacent tokens to determine
    when they should form cognitive chunks.
    """

    def __init__(self, dim: int, max_chunk_size: int = 5, fixed_bind_threshold: float = 0.5):
        super().__init__()
        self.dim = dim
        self.max_chunk_size = max_chunk_size

        self.bind_key = nn.Linear(dim, dim // 2)
        self.bind_query = nn.Linear(dim, dim // 2)
        self.register_buffer('bind_threshold', torch.tensor(fixed_bind_threshold))

        # Chunk energy component is now removed from the binding_potential calculation
        # but the layers are kept in case we want to reintroduce it later.
        self.chunk_energy_module = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        if T <= 1:
            binding_mask = torch.zeros(B, T, device=x.device, dtype=torch.float)
            binding_potential_padded = torch.zeros(B, T - 1 if T > 1 else 0, device=x.device, dtype=torch.float)
            return binding_mask, binding_potential_padded

        keys = self.bind_key(x[:, :-1])
        queries = self.bind_query(x[:, 1:])
        binding_strength = torch.sigmoid(
            (keys * queries).sum(dim=-1) / math.sqrt(self.dim // 2)
        )

        # --- Modification: Use binding_strength directly as binding_potential ---
        # paired_features = torch.cat([x[:, :-1], x[:, 1:]], dim=-1)
        # chunk_energy_values = torch.sigmoid(self.chunk_energy_module(paired_features)).squeeze(-1)
        # binding_potential = (binding_strength + chunk_energy_values) / 2.0 # Old: Averaging

        binding_potential = binding_strength  # New: Directly use binding_strength

        binding_mask = torch.zeros(B, T, device=x.device, dtype=torch.float)
        binding_mask[:, 1:] = (binding_potential > self.bind_threshold).float()

        return binding_mask, binding_potential


class TemporalScale(nn.Module):
    """
    Represents one temporal scale in the hierarchy.
    Processes information at a specific temporal resolution, potentially with sparse updates.
    """

    def __init__(self, dim: int, num_heads: int, update_frequency: int):
        super().__init__()
        self.dim = dim
        self.update_frequency = update_frequency
        self.step_count = 0

        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.register_buffer('state_memory', None, persistent=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        self.step_count += 1
        if self.update_frequency > 1 and (
                self.step_count - 1) % self.update_frequency != 0 and self.state_memory is not None:
            if self.state_memory.size(0) == x.size(0):
                return self.state_memory

        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        if self.update_frequency > 1:
            self.state_memory = x.clone().detach()
        return x


class HierarchicalBinder(nn.Module):
    """
    Manages the binding and unbinding of tokens into hierarchical chunks
    across multiple temporal scales. Also handles downward influence.
    """

    def __init__(self, dim: int, num_scales: int = 3, fixed_bind_threshold: float = 0.5):
        super().__init__()
        self.dim = dim
        self.num_scales = num_scales

        self.binding_fields = nn.ModuleList([
            BindingField(dim, max_chunk_size=2 ** (i + 1), fixed_bind_threshold=fixed_bind_threshold)
            for i in range(num_scales - 1)
        ])

        self.aggregators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(),
                nn.Linear(dim, dim)
            )
            for _ in range(num_scales - 1)
        ])

        self.downward_influence = nn.ModuleList([
            nn.Linear(dim, dim)
            for _ in range(num_scales - 1)
        ])

    def aggregate_chunks(self, x: torch.Tensor, binding_mask: torch.Tensor,
                         scale_idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T_lower, D = x.shape
        if T_lower == 0:
            return torch.zeros(B, 0, D, device=x.device), torch.ones(B, 0, device=x.device, dtype=torch.bool)

        chunk_starts = torch.zeros_like(binding_mask, dtype=torch.bool)
        chunk_starts[:, 0] = True
        if T_lower > 1:
            chunk_starts[:, 1:] = (1 - binding_mask[:, 1:]).bool()

        aggregated_batches = []
        chunk_lengths = []

        for b in range(B):
            batch_chunks = []
            current_chunk_tokens = []
            for t in range(T_lower):
                if chunk_starts[b, t] and current_chunk_tokens:
                    chunk_tensor = torch.stack(current_chunk_tokens)
                    chunk_repr_mean = chunk_tensor.mean(dim=0)
                    chunk_repr = self.aggregators[scale_idx](chunk_repr_mean)
                    batch_chunks.append(chunk_repr)
                    current_chunk_tokens = []
                current_chunk_tokens.append(x[b, t])
            if current_chunk_tokens:
                chunk_tensor = torch.stack(current_chunk_tokens)
                chunk_repr_mean = chunk_tensor.mean(dim=0)
                chunk_repr = self.aggregators[scale_idx](chunk_repr_mean)
                batch_chunks.append(chunk_repr)

            if batch_chunks:
                aggregated_batches.append(torch.stack(batch_chunks))
                chunk_lengths.append(len(batch_chunks))
            else:
                aggregated_batches.append(torch.empty(0, D, device=x.device))
                chunk_lengths.append(0)

        if not any(cl > 0 for cl in chunk_lengths):
            return torch.zeros(B, 0, D, device=x.device), torch.ones(B, 0, device=x.device, dtype=torch.bool)

        max_chunks = max(chunk_lengths)
        padded_aggregated_chunks = torch.zeros(B, max_chunks, D, device=x.device)
        padding_mask = torch.ones(B, max_chunks, device=x.device, dtype=torch.bool)

        for b, agg_b in enumerate(aggregated_batches):
            if agg_b.size(0) > 0:
                padded_aggregated_chunks[b, :agg_b.size(0)] = agg_b
                padding_mask[b, :agg_b.size(0)] = False
        return padded_aggregated_chunks, padding_mask

    def apply_downward_influence(self, higher_scale_repr: torch.Tensor,
                                 lower_scale_repr: torch.Tensor,
                                 scale_idx: int,
                                 chunk_map: Optional[List[List[Tuple[int, int]]]] = None):
        B, T_higher, D_high = higher_scale_repr.shape
        B, T_lower, D_low = lower_scale_repr.shape

        if T_higher == 0 or T_lower == 0:
            return lower_scale_repr

        influence_signal = self.downward_influence[scale_idx](higher_scale_repr)
        if chunk_map is not None:
            influence_expanded = torch.zeros_like(lower_scale_repr)
            for b in range(B):
                for i_chunk_higher in range(T_higher):
                    if i_chunk_higher < len(chunk_map[b]):
                        start_idx_lower, end_idx_lower = chunk_map[b][i_chunk_higher]
                        valid_end_idx = min(end_idx_lower + 1, T_lower)
                        if start_idx_lower < valid_end_idx:
                            influence_expanded[b, start_idx_lower:valid_end_idx] = influence_signal[
                                b, i_chunk_higher].unsqueeze(0)
        else:
            if T_higher > 0 and T_lower > 0:
                expansion_factor = (T_lower + T_higher - 1) // T_higher
                influence_expanded_temp = influence_signal.repeat_interleave(expansion_factor, dim=1)
                if influence_expanded_temp.size(1) > T_lower:
                    influence_expanded = influence_expanded_temp[:, :T_lower, :]
                elif influence_expanded_temp.size(1) < T_lower:
                    pad_size = T_lower - influence_expanded_temp.size(1)
                    influence_expanded = F.pad(influence_expanded_temp, (0, 0, 0, pad_size, 0, 0))
                else:
                    influence_expanded = influence_expanded_temp
            else:
                influence_expanded = torch.zeros_like(lower_scale_repr)
        gate = torch.sigmoid(influence_expanded)
        influenced_lower_scale_repr = lower_scale_repr * gate + lower_scale_repr
        return influenced_lower_scale_repr


class TemporalHierarchicalTransformer(nn.Module):
    """
    Transformer that processes information at multiple temporal scales
    with dynamic binding/unbinding of tokens into cognitive chunks.
    """

    def __init__(
            self,
            vocab_size: int,
            dim: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            num_scales: int = 3,
            max_seq_len: int = 512,
            update_frequencies: Optional[List[int]] = None,
            fixed_bind_threshold: float = 0.5
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_scales = num_scales

        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.binder = HierarchicalBinder(dim, num_scales, fixed_bind_threshold=fixed_bind_threshold)

        if update_frequencies is None:
            update_frequencies = [2 ** i for i in range(num_scales)]
        elif len(update_frequencies) != num_scales:
            raise ValueError("Length of update_frequencies must match num_scales")

        self.scales = nn.ModuleList([
            TemporalScale(dim, num_heads, update_frequencies[i])
            for i in range(num_scales)
        ])

        self.cross_scale_attention_up = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads // 2, batch_first=True)
            for _ in range(num_scales - 1)
        ])

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(dim, num_heads)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, x_indices: torch.Tensor) -> torch.Tensor:
        B, T = x_indices.shape
        if T > self.pos_emb.num_embeddings:
            raise ValueError(
                f"Input sequence length {T} exceeds max_seq_len {self.pos_emb.num_embeddings}. "
                "Truncate input or increase max_seq_len."
            )

        tok_emb = self.token_emb(x_indices)
        pos_indices = torch.arange(T, device=x_indices.device).unsqueeze(0).repeat(B, 1)
        pos_emb = self.pos_emb(pos_indices)
        current_repr = tok_emb + pos_emb

        scale_representations = []
        all_chunk_maps = []
        current_padding_mask = (
                    x_indices == self.token_emb.padding_idx) if self.token_emb.padding_idx is not None else None

        for scale_idx in range(self.num_scales):
            processed_repr = self.scales[scale_idx](current_repr, mask=current_padding_mask)
            scale_representations.append(processed_repr)

            if scale_idx < self.num_scales - 1:
                binding_mask, _ = self.binder.binding_fields[scale_idx](processed_repr)
                aggregated_chunks, higher_scale_padding_mask = self.binder.aggregate_chunks(
                    processed_repr,
                    binding_mask,
                    scale_idx
                )
                current_repr = aggregated_chunks
                current_padding_mask = higher_scale_padding_mask
                if current_repr.size(1) == 0:
                    for _ in range(scale_idx + 1, self.num_scales):
                        scale_representations.append(torch.zeros(B, 0, self.binder.dim, device=x_indices.device))
                    break
            else:
                pass

        influenced_representations = list(scale_representations)
        for scale_idx_lower in range(self.num_scales - 2, -1, -1):
            higher_scale_level = scale_idx_lower + 1
            higher_repr = influenced_representations[higher_scale_level]
            lower_repr = influenced_representations[scale_idx_lower]

            if higher_repr.size(1) > 0 and lower_repr.size(1) > 0:
                influenced_lower = self.binder.apply_downward_influence(
                    higher_repr,
                    lower_repr,
                    scale_idx_lower,
                )
                influenced_representations[scale_idx_lower] = influenced_lower

        final_unified_repr = influenced_representations[0]
        final_padding_mask = (
                    x_indices == self.token_emb.padding_idx) if self.token_emb.padding_idx is not None else None
        for layer in self.transformer_layers:
            final_unified_repr = layer(final_unified_repr, mask=final_padding_mask)

        output = self.ln_f(final_unified_repr)
        logits = self.lm_head(output)
        return logits


class TransformerLayer(nn.Module):
    """Standard transformer layer (self-attention + FFN)."""

    def __init__(self, dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 4, dim)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask, need_weights=False)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff_out = self.ff(x)
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x


class CompositionalReasoningDataset:
    """
    Base dataset for tasks requiring understanding of hierarchical binding.
    """

    def __init__(self, vocab_size: int = 1000, num_examples: int = 10000):
        self.vocab_size = vocab_size
        self.num_examples = num_examples
        self.PAD = 0
        self.START = 1
        self.END = 2
        self.SEP = 3
        self.EQUALS = 4
        self.WHAT = 50
        self.WHO = 51
        self.Q_MARK = 52
        self.concepts = {
            'red_car': (10, 11), 'blue_sky': (12, 13), 'green_tree': (14, 15),
            'big_house': (16, 17), 'small_cat': (18, 19), 'fast_train': (20, 21),
            'old_book': (22, 23), 'new_phone': (24, 25),
        }
        self.concept_token_values = set(sum([list(v) for v in self.concepts.values()], []))
        self.relations = {'has': 30, 'is': 31, 'likes': 32, 'sees': 33, 'near': 34}
        self.entities = {'john': 40, 'mary': 41, 'bob': 42, 'alice': 43}
        self.entity_token_values = set(self.entities.values())

    def _get_concept_tokens_from_key(self, concept_key: str) -> Tuple[int, ...]:
        return self.concepts[concept_key]

    def generate_binding_task(self) -> Tuple[List[int], List[int], Dict]:
        concept_keys = list(self.concepts.keys())
        c1_key, c2_key = np.random.choice(concept_keys, 2, replace=False)
        relation_key = np.random.choice(list(self.relations.keys()))
        relation_token = self.relations[relation_key]
        tokens_c1 = list(self._get_concept_tokens_from_key(c1_key))
        tokens_c2 = list(self._get_concept_tokens_from_key(c2_key))
        statement_tokens = [self.START] + tokens_c1 + [relation_token] + tokens_c2 + [self.SEP]
        question_tokens = [self.WHAT, relation_token] + tokens_c2 + [self.Q_MARK]
        answer_tokens = tokens_c1 + [self.END]
        full_sequence = statement_tokens + question_tokens + answer_tokens
        input_seq = full_sequence[:-1]
        target_seq = full_sequence[1:]
        c1_start_idx = 1
        c1_end_idx = c1_start_idx + len(tokens_c1) - 1
        c2_start_idx = c1_end_idx + 1 + 1
        c2_end_idx = c2_start_idx + len(tokens_c2) - 1
        metadata = {
            'task_type': 'binding', 'concept1': c1_key, 'concept2': c2_key, 'relation': relation_key,
            'binding_spans': [(c1_start_idx, c1_end_idx), (c2_start_idx, c2_end_idx)],
            'answer_span_in_target': (len(statement_tokens) + len(question_tokens) - 1, len(target_seq) - 2)
        }
        return input_seq, target_seq, metadata

    def generate_compositional_task(self) -> Tuple[List[int], List[int], Dict]:
        entities_keys = list(self.entities.keys())
        e1_key, e2_key = np.random.choice(entities_keys, 2, replace=False)
        concept_keys = list(self.concepts.keys())
        c1_key, c2_key = np.random.choice(concept_keys, 2, replace=False)
        tokens_e1 = self.entities[e1_key];
        tokens_e2 = self.entities[e2_key]
        tokens_c1 = list(self._get_concept_tokens_from_key(c1_key))
        tokens_c2 = list(self._get_concept_tokens_from_key(c2_key))
        stmt1_tokens = [tokens_e1, self.relations['has']] + tokens_c1 + [self.SEP]
        stmt2_tokens = [tokens_e2, self.relations['has']] + tokens_c2 + [self.SEP]
        question_tokens = [self.WHO, self.relations['has']] + tokens_c1 + [self.Q_MARK]
        answer_tokens = [tokens_e1, self.END]
        full_sequence = [self.START] + stmt1_tokens + stmt2_tokens + question_tokens + answer_tokens
        input_seq = full_sequence[:-1]
        target_seq = full_sequence[1:]
        c1_stmt1_start_idx = 1 + 1 + 1
        c1_stmt1_end_idx = c1_stmt1_start_idx + len(tokens_c1) - 1
        c2_stmt2_start_idx = 1 + len(stmt1_tokens) + 1 + 1
        c2_stmt2_end_idx = c2_stmt2_start_idx + len(tokens_c2) - 1
        c1_q_start_idx = 1 + len(stmt1_tokens) + len(stmt2_tokens) + 1 + 1
        c1_q_end_idx = c1_q_start_idx + len(tokens_c1) - 1
        metadata = {
            'task_type': 'compositional', 'entities': (e1_key, e2_key), 'concepts': (c1_key, c2_key),
            'query_concept': c1_key, 'answer_entity': e1_key,
            'binding_spans': [(c1_stmt1_start_idx, c1_stmt1_end_idx), (c2_stmt2_start_idx, c2_stmt2_end_idx),
                              (c1_q_start_idx, c1_q_end_idx)],
            'answer_span_in_target': (len(full_sequence) - 1 - 1 - 1, len(target_seq) - 2)
        }
        return input_seq, target_seq, metadata

    def generate_batch(self, batch_size: int, task_type: str = 'mixed', max_len: Optional[int] = None):
        inputs_list, targets_list, metadata_list = [], [], []
        for _ in range(batch_size):
            if task_type == 'binding' or (task_type == 'mixed' and np.random.random() < 0.5):
                inp, tgt, meta = self.generate_binding_task()
            else:
                inp, tgt, meta = self.generate_compositional_task()
            inputs_list.append(torch.tensor(inp, dtype=torch.long))
            targets_list.append(torch.tensor(tgt, dtype=torch.long))
            metadata_list.append(meta)

        current_max_len_in = max(len(s) for s in inputs_list)
        current_max_len_tgt = max(len(s) for s in targets_list)
        effective_max_len = max(current_max_len_in, current_max_len_tgt) if max_len is None else max_len

        if max_len is not None:
            inputs_list = [s[:effective_max_len] for s in inputs_list]
            targets_list = [s[:effective_max_len] for s in targets_list]

        padded_inputs = torch.full((batch_size, effective_max_len), self.PAD, dtype=torch.long)
        padded_targets = torch.full((batch_size, effective_max_len), self.PAD, dtype=torch.long)

        for i in range(batch_size):
            len_in = len(inputs_list[i]);
            padded_inputs[i, :len_in] = inputs_list[i]
            len_tgt = len(targets_list[i]);
            padded_targets[i, :len_tgt] = targets_list[i]
            if self.PAD != IGNORE_INDEX:
                padded_targets[i, len_tgt:] = IGNORE_INDEX
        return padded_inputs, padded_targets, metadata_list


class HardCompositionalReasoningDataset(CompositionalReasoningDataset):
    def __init__(self, vocab_size: int = 2000, num_examples: int = 20000):
        super().__init__(vocab_size, num_examples)
        self.fillers = list(range(60, 70))
        self.distractors_adj = list(range(70, 80))
        self.distractors_noun = list(range(80, 90))
        self.UNANSWERABLE = 91
        self.concepts.update({
            'dark_red_car': (100, 10, 11), 'bright_blue_sky': (101, 12, 13),
            'old_green_tree': (102, 14, 15), 'fluffy_small_cat': (103, 18, 19)
        })
        new_concept_tokens = set(
            sum([list(v) for k, v in self.concepts.items() if k.startswith(('dark', 'bright', 'old', 'fluffy'))], []))
        self.concept_token_values.update(new_concept_tokens)
        all_tokens_flat = set()
        for token_collection in [
            {self.PAD, self.START, self.END, self.SEP, self.EQUALS, self.WHAT, self.WHO, self.Q_MARK,
             self.UNANSWERABLE},
            self.concept_token_values, set(self.relations.values()), self.entity_token_values,
            set(self.fillers), set(self.distractors_adj), set(self.distractors_noun)
        ]: all_tokens_flat.update(token_collection)
        max_token_val = max(all_tokens_flat) if all_tokens_flat else 0
        if self.vocab_size <= max_token_val:
            print(
                f"Warning: vocab_size {self.vocab_size} is too small for HardDataset. Max token value used: {max_token_val}. Adjusting vocab_size.")
            self.vocab_size = max_token_val + 1

    def _get_concept_tokens_from_key(self, concept_key: str) -> Tuple[int, ...]:
        base_tokens = list(self.concepts[concept_key])
        if len(base_tokens) > 1 and np.random.rand() < 0.3:
            base_tokens = [base_tokens[0], np.random.choice(self.fillers)] + base_tokens[1:]
        if np.random.rand() < 0.3:
            base_tokens = [np.random.choice(self.distractors_adj)] + base_tokens
        return tuple(base_tokens)

    def generate_binding_task(self) -> Tuple[List[int], List[int], Dict]:
        return super().generate_binding_task()

    def generate_compositional_task(self) -> Tuple[List[int], List[int], Dict]:
        num_facts = np.random.randint(2, 5)
        story_entities_keys = np.random.choice(list(self.entities.keys()), num_facts, replace=True)
        story_concepts_keys = np.random.choice(list(self.concepts.keys()), num_facts, replace=True)
        statement_tokens_all, current_binding_spans, current_offset = [], [], 1

        for i in range(num_facts):
            e_key, c_key = story_entities_keys[i], story_concepts_keys[i]
            tokens_e = self.entities[e_key]
            tokens_c = list(self._get_concept_tokens_from_key(c_key))
            stmt = [tokens_e, self.relations['has']] + tokens_c + [self.SEP]
            statement_tokens_all.extend(stmt)
            c_start_idx = current_offset + 1 + 1
            c_end_idx = c_start_idx + len(tokens_c) - 1
            current_binding_spans.append((c_start_idx, c_end_idx))
            current_offset += len(stmt)

        query_fact_idx = np.random.randint(num_facts)
        query_entity_key = story_entities_keys[query_fact_idx]
        query_concept_key = story_concepts_keys[query_fact_idx]
        tokens_query_c = list(self._get_concept_tokens_from_key(query_concept_key))
        question_is_unanswerable = np.random.rand() < 0.2
        actual_answer_entity = None

        if question_is_unanswerable:
            if np.random.rand() < 0.5:
                other_concept_keys = [k for k in self.concepts.keys() if k != query_concept_key]
                if other_concept_keys:
                    asked_c_key = np.random.choice(other_concept_keys)
                    tokens_query_c_unans = list(self._get_concept_tokens_from_key(asked_c_key))
                    question_tokens = [self.WHO, self.relations['has']] + tokens_query_c_unans + [self.Q_MARK]
                else:
                    question_is_unanswerable = False
            else:
                question_tokens = [self.WHO, self.relations['has']] + tokens_query_c + [self.Q_MARK]
            if question_is_unanswerable: answer_tokens = [self.UNANSWERABLE, self.END]

        if not question_is_unanswerable:
            question_tokens = [self.WHO, self.relations['has']] + tokens_query_c + [self.Q_MARK]
            answer_tokens = [self.entities[query_entity_key], self.END]
            actual_answer_entity = query_entity_key

        q_c_start_idx = current_offset + 1 + 1
        q_c_end_idx = q_c_start_idx + len(tokens_query_c if not (
                    question_is_unanswerable and 'tokens_query_c_unans' in locals()) else tokens_query_c_unans) - 1
        current_binding_spans.append((q_c_start_idx, q_c_end_idx))

        full_sequence = [self.START] + statement_tokens_all + question_tokens + answer_tokens
        input_seq = full_sequence[:-1];
        target_seq = full_sequence[1:]
        metadata = {
            'task_type': 'hard_compositional', 'num_facts': num_facts, 'query_concept': query_concept_key,
            'answer_entity': actual_answer_entity, 'is_unanswerable': question_is_unanswerable,
            'binding_spans': current_binding_spans,
            'answer_span_in_target': (len(full_sequence) - 1 - 1 - 1, len(target_seq) - 2)
        }
        return input_seq, target_seq, metadata


class BindingAnalyzer:
    """Analyzes binding behavior during training."""

    def __init__(self, fixed_model_bind_threshold: float = 0.5):
        self.binding_strengths_at_concepts = defaultdict(list)
        self.binding_strengths_elsewhere = defaultdict(list)
        self.model_accuracy = defaultdict(list)
        self.task_specific_accuracy = defaultdict(list)
        self.losses = defaultdict(list)
        self.fixed_model_bind_threshold = fixed_model_bind_threshold

    def analyze_step(self, model: nn.Module,
                     inputs: torch.Tensor, metadata_batch: List[Dict],
                     loss_val: float, model_key: str = "temporal"):
        model.eval()
        with torch.no_grad():
            is_temporal_model = isinstance(model, TemporalHierarchicalTransformer)
            binding_potentials_t_minus_1 = None

            if is_temporal_model:
                if inputs.size(1) > model.pos_emb.num_embeddings:
                    print(
                        f"Warning (Analyzer): Input seq length {inputs.size(1)} > pos_emb {model.pos_emb.num_embeddings}")
                elif hasattr(model, 'binder') and model.binder.binding_fields:
                    tok_emb = model.token_emb(inputs)
                    pos_idx = torch.arange(inputs.size(1), device=inputs.device).unsqueeze(0).repeat(inputs.size(0), 1)
                    pos_emb = model.pos_emb(pos_idx)
                    x_emb = tok_emb + pos_emb
                    _, binding_potentials_t_minus_1 = model.binder.binding_fields[0](x_emb)

            if binding_potentials_t_minus_1 is not None:
                step_concept_strengths, step_elsewhere_strengths = [], []
                for i, meta_item in enumerate(metadata_batch):
                    is_concept_binding_loc = torch.zeros(binding_potentials_t_minus_1.size(1), dtype=torch.bool,
                                                         device=inputs.device)
                    if 'binding_spans' in meta_item:
                        for concept_start_idx, concept_end_idx in meta_item['binding_spans']:
                            if concept_start_idx <= concept_end_idx - 1:
                                valid_start = max(0, concept_start_idx)
                                valid_end = min(binding_potentials_t_minus_1.size(1), concept_end_idx)
                                if valid_start < valid_end:
                                    is_concept_binding_loc[valid_start: valid_end] = True
                    if torch.any(is_concept_binding_loc):
                        step_concept_strengths.extend(binding_potentials_t_minus_1[i, is_concept_binding_loc].tolist())

                    is_elsewhere_loc = ~is_concept_binding_loc
                    padding_idx_val = getattr(model.token_emb, 'padding_idx', 0)
                    actual_len_item = inputs[i].ne(padding_idx_val).sum().item()
                    valid_seq_len_for_binding_potential = actual_len_item - 1
                    if valid_seq_len_for_binding_potential < is_elsewhere_loc.size(
                            0) and valid_seq_len_for_binding_potential >= 0:
                        is_elsewhere_loc[valid_seq_len_for_binding_potential:] = False
                    if torch.any(is_elsewhere_loc):
                        step_elsewhere_strengths.extend(binding_potentials_t_minus_1[i, is_elsewhere_loc].tolist())

                self.binding_strengths_at_concepts[model_key].append(
                    np.mean(step_concept_strengths) if step_concept_strengths else 0.0)
                self.binding_strengths_elsewhere[model_key].append(
                    np.mean(step_elsewhere_strengths) if step_elsewhere_strengths else 0.0)

            self.losses[model_key].append(loss_val)
        model.train()

    def record_accuracy(self, acc_val: float, model_key: str, task_key: Optional[str] = None):
        if task_key:
            self.task_specific_accuracy[f"{model_key}_{task_key}"].append(acc_val)
        else:
            self.model_accuracy[model_key].append(acc_val)

    def plot_analysis(self, save_path: str = 'binding_analysis.png'):
        num_plots = 0
        if any(self.binding_strengths_at_concepts.values()): num_plots += 1
        if any(self.model_accuracy.values()) or any(self.task_specific_accuracy.values()): num_plots += 1
        if any(self.losses.values()): num_plots += 1
        if num_plots == 0: print("No data to plot for BindingAnalyzer."); return

        fig, axes = plt.subplots(num_plots, 1, figsize=(10, num_plots * 4), squeeze=False)
        plot_idx = 0

        if any(self.binding_strengths_at_concepts.values()):
            ax = axes[plot_idx, 0]
            for mk in self.binding_strengths_at_concepts:
                if self.binding_strengths_at_concepts[mk]:
                    ax.plot(self.binding_strengths_at_concepts[mk], label=f'{mk} - Concept Bindings')
            for mk in self.binding_strengths_elsewhere:
                if self.binding_strengths_elsewhere[mk]:
                    ax.plot(self.binding_strengths_elsewhere[mk], label=f'{mk} - Elsewhere Bindings', linestyle=':')
            ax.axhline(y=self.fixed_model_bind_threshold, color='r', linestyle='--',
                       label=f'Fixed Bind Threshold ({self.fixed_model_bind_threshold:.2f})')
            ax.set_title('Average Binding Potential Over Training');
            ax.set_xlabel('Analysis Step');
            ax.set_ylabel('Binding Potential');
            ax.legend();
            plot_idx += 1

        if any(self.model_accuracy.values()) or any(self.task_specific_accuracy.values()):
            ax = axes[plot_idx, 0]
            for k, v in self.model_accuracy.items():
                if v: ax.plot(v, label=f'{k} (Overall Acc)')
            for k, v in self.task_specific_accuracy.items():
                if v: ax.plot(v, label=f'{k} (Task Acc)', linestyle='--')
            ax.set_title('Model Accuracy');
            ax.set_xlabel('Analysis Step');
            ax.set_ylabel('Accuracy');
            ax.legend();
            plot_idx += 1

        if any(self.losses.values()):
            ax = axes[plot_idx, 0]
            for k, v in self.losses.items():
                if v: ax.plot(v, label=f'{k} Loss')
            ax.set_title('Training Loss');
            ax.set_xlabel('Training Step');
            ax.set_ylabel('Loss');
            ax.legend();
            plot_idx += 1
        plt.tight_layout();
        plt.savefig(save_path);
        print(f"Binding analysis plot saved to {save_path}");
        plt.close()


class StandardTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 512, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 512, dropout_rate: float = 0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layers = nn.ModuleList([TransformerLayer(dim, num_heads, dropout_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, x_indices: torch.Tensor) -> torch.Tensor:
        B, T = x_indices.shape
        if T > self.pos_emb.num_embeddings:
            raise ValueError(f"Input len {T} > max_seq_len {self.pos_emb.num_embeddings}")
        tok_emb = self.token_emb(x_indices);
        pos_idx = torch.arange(T, device=x_indices.device).unsqueeze(0).repeat(B, 1)
        pos_emb = self.pos_emb(pos_idx);
        x = self.dropout(tok_emb + pos_emb)
        padding_mask = (x_indices == self.token_emb.padding_idx) if self.token_emb.padding_idx is not None else None
        for layer in self.layers: x = layer(x, mask=padding_mask)
        return self.lm_head(self.ln_f(x))


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_idx: int) -> float:
    preds = logits.argmax(dim=-1);
    mask = (targets != ignore_idx)
    correct_preds = (preds == targets) & mask
    return correct_preds.sum().item() / mask.sum().item() if mask.sum().item() > 0 else 0.0


def train_and_compare():
    # Hyperparameters
    fixed_bind_threshold_value = BIND_THRESHOLD

    vocab_size_ds = 150
    dim = 128
    num_layers_main = 2
    num_heads = 4
    num_scales_temporal = 3
    batch_size = 16
    learning_rate = 1e-4
    num_steps = 1000
    analysis_every_n_steps = 50
    max_seq_len_model = 64

    dataset = HardCompositionalReasoningDataset(vocab_size=vocab_size_ds)
    actual_vocab_size = dataset.vocab_size
    print(f"Using dataset with effective vocab_size: {actual_vocab_size}")

    temporal_model = TemporalHierarchicalTransformer(
        vocab_size=actual_vocab_size, dim=dim, num_layers=num_layers_main, num_heads=num_heads,
        num_scales=num_scales_temporal, max_seq_len=max_seq_len_model,
        fixed_bind_threshold=BIND_THRESHOLD
    ).to(DEVICE)

    baseline_model = StandardTransformer(
        vocab_size=actual_vocab_size, dim=dim, num_layers=num_layers_main * num_scales_temporal,
        num_heads=num_heads, max_seq_len=max_seq_len_model
    ).to(DEVICE)

    analyzer = BindingAnalyzer(fixed_model_bind_threshold=BIND_THRESHOLD)

    temporal_opt = torch.optim.Adam(temporal_model.parameters(), lr=learning_rate)
    baseline_opt = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)
    gradient_clip_value = 1.0

    print(
        f"Training on {DEVICE} with Max Seq Len: {max_seq_len_model}, Fixed Bind Threshold: {BIND_THRESHOLD}")
    print("=" * 70)

    for step in range(num_steps):
        inputs, targets, metadata = dataset.generate_batch(batch_size, task_type='mixed', max_len=max_seq_len_model)
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        targets_for_loss = targets.clone()
        targets_for_loss[targets == dataset.PAD] = IGNORE_INDEX
        if hasattr(dataset, 'UNANSWERABLE'): targets_for_loss[targets == dataset.UNANSWERABLE] = IGNORE_INDEX

        temporal_model.train();
        temporal_opt.zero_grad()
        temporal_logits = temporal_model(inputs)
        temporal_loss = F.cross_entropy(temporal_logits.view(-1, actual_vocab_size), targets_for_loss.view(-1),
                                        ignore_index=IGNORE_INDEX)
        temporal_loss.backward();
        torch.nn.utils.clip_grad_norm_(temporal_model.parameters(), gradient_clip_value);
        temporal_opt.step()

        baseline_model.train();
        baseline_opt.zero_grad()
        baseline_logits = baseline_model(inputs)
        baseline_loss = F.cross_entropy(baseline_logits.view(-1, actual_vocab_size), targets_for_loss.view(-1),
                                        ignore_index=IGNORE_INDEX)
        baseline_loss.backward();
        torch.nn.utils.clip_grad_norm_(baseline_model.parameters(), gradient_clip_value);
        baseline_opt.step()

        if step % analysis_every_n_steps == 0 or step == num_steps - 1:
            temporal_model.eval();
            baseline_model.eval()
            analyzer.analyze_step(temporal_model, inputs, metadata, temporal_loss.item(), "Temporal")
            analyzer.losses["Baseline"].append(baseline_loss.item())

            with torch.no_grad():
                temporal_acc_mixed = compute_accuracy(temporal_logits, targets, IGNORE_INDEX)
                baseline_acc_mixed = compute_accuracy(baseline_logits, targets, IGNORE_INDEX)
                analyzer.record_accuracy(temporal_acc_mixed, "Temporal", "mixed_batch")
                analyzer.record_accuracy(baseline_acc_mixed, "Baseline", "mixed_batch")

                bind_inputs, bind_targets, _ = dataset.generate_batch(batch_size, task_type='binding',
                                                                      max_len=max_seq_len_model)
                bind_inputs, bind_targets = bind_inputs.to(DEVICE), bind_targets.to(DEVICE)
                temporal_acc_binding = compute_accuracy(temporal_model(bind_inputs), bind_targets, IGNORE_INDEX)
                baseline_acc_binding = compute_accuracy(baseline_model(bind_inputs), bind_targets, IGNORE_INDEX)
                analyzer.record_accuracy(temporal_acc_binding, "Temporal", "binding_task")
                analyzer.record_accuracy(baseline_acc_binding, "Baseline", "binding_task")

            print(f"Step {step:4d} | T_Loss: {temporal_loss.item():.4f} | B_Loss: {baseline_loss.item():.4f} | "
                  f"T_Acc_Mix: {temporal_acc_mixed:.4f} | B_Acc_Mix: {baseline_acc_mixed:.4f} | "
                  f"T_Acc_Bind: {temporal_acc_binding:.4f} | B_Acc_Bind: {baseline_acc_binding:.4f}")
            if hasattr(temporal_model,
                       'binder') and temporal_model.binder.binding_fields and analyzer.binding_strengths_at_concepts.get(
                    "Temporal"):
                print(
                    f"          | T_Concept_Bind_Strength: {analyzer.binding_strengths_at_concepts['Temporal'][-1]:.4f}")

    print("\n" + "=" * 70 + "\nTraining Complete! Generating analysis plots...")
    analyzer.plot_analysis(save_path='binding_analysis_revised.png')

    print("\nDemonstrating binding behavior on one example (Temporal Model):")
    temporal_model.eval()
    inputs_ex, _, metadata_ex_list = dataset.generate_batch(1, task_type='binding', max_len=max_seq_len_model)
    inputs_ex = inputs_ex.to(DEVICE);
    metadata_ex = metadata_ex_list[0]

    with torch.no_grad():
        if hasattr(temporal_model, 'binder') and temporal_model.binder.binding_fields:
            tok_emb = temporal_model.token_emb(inputs_ex)
            pos_idx = torch.arange(inputs_ex.size(1), device=inputs_ex.device).unsqueeze(0)
            pos_emb = temporal_model.pos_emb(pos_idx)
            x_emb = tok_emb + pos_emb
            current_fixed_threshold = temporal_model.binder.binding_fields[0].bind_threshold.item()
            binding_mask_out, binding_potential_out = temporal_model.binder.binding_fields[0](x_emb)

            print(f"\nInput tokens: {inputs_ex[0].tolist()}")
            print(f"Binding potentials (raw): {binding_potential_out[0].tolist()}")
            print(f"Binding mask ( > threshold {current_fixed_threshold:.2f} ): {binding_mask_out[0, 1:].tolist()}")
            print(f"Expected concept spans (token indices in input): {metadata_ex.get('binding_spans', 'N/A')}")
        else:
            print("Temporal model does not have binding fields to demonstrate.")
    return temporal_model, baseline_model, analyzer


