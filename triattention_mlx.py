"""
TriAttention MLX — Apple Silicon KV Cache Compression
=====================================================
Self-contained port of TriAttention for mlx-lm models on Apple Silicon.

Based on: https://github.com/WeianMao/triattention (Apache 2.0)
MLX port: @DeadByDawn101

Usage:
    from triattention_mlx import TriAttentionCompressor

    compressor = TriAttentionCompressor(kv_budget=512)
    # After each decode step, call:
    kv_cache = compressor.step(kv_cache, is_prefill=True/False)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import numpy as np


@dataclass
class TriAttentionConfig:
    """Configuration for TriAttention KV compression."""
    kv_budget: int = 512
    """Maximum number of KV pairs to keep per layer."""

    divide_length: int = 128
    """Compress every N decode steps."""

    prefill_pin: bool = True
    """Always preserve prefill (prompt) tokens."""

    head_dim: int = 256
    """Attention head dimension."""

    rope_theta: float = 10000.0
    """RoPE base frequency."""


def build_inv_freq(head_dim: int, rope_theta: float) -> mx.array:
    """Build inverse frequencies for RoPE."""
    i = mx.arange(0, head_dim // 2, dtype=mx.float32)
    return 1.0 / (rope_theta ** (2 * i / head_dim))


def invert_rope(k: mx.array, positions: mx.array, inv_freq: mx.array) -> mx.array:
    """
    Remove RoPE rotation from key vectors to get pre-RoPE representation.

    k: [seq_len, head_dim]
    positions: [seq_len]
    inv_freq: [head_dim/2]

    Returns: k_pre_rope [seq_len, head_dim]
    """
    theta = mx.outer(positions.astype(mx.float32), inv_freq)
    cos_t = mx.cos(theta)
    sin_t = mx.sin(theta)

    k = k.astype(mx.float32)
    k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]:]

    k1_orig = k1 * cos_t + k2 * sin_t
    k2_orig = -k1 * sin_t + k2 * cos_t

    return mx.concatenate([k1_orig, k2_orig], axis=-1)


def score_keys_norm(k: mx.array) -> mx.array:
    """
    Score keys using norm-only scoring (no trig stats needed).
    Higher norm = more important to keep.

    k: [seq_len, head_dim]
    Returns: scores [seq_len]
    """
    return mx.sqrt(mx.sum(k.astype(mx.float32) ** 2, axis=-1) + 1e-8)


class TriAttentionCompressor:
    """
    TriAttention KV cache compressor for MLX models.

    Integrates with mlx-lm's KV cache by compressing it when it exceeds
    the configured budget. Uses norm-only scoring (no calibration stats needed).
    """

    def __init__(self, config: TriAttentionConfig):
        self.config = config
        self.inv_freq = build_inv_freq(config.head_dim, config.rope_theta)

        self.cache_positions: List[int] = []
        self.absolute_position: int = 0
        self.prefix_length: int = 0
        self.step_count: int = 0
        self.compress_count: int = 0

    def reset(self):
        """Reset for new generation."""
        self.cache_positions = []
        self.absolute_position = 0
        self.prefix_length = 0
        self.step_count = 0

    def should_compress(self, cache_len: int) -> bool:
        """Check if we should compress now."""
        effective = cache_len
        if self.config.prefill_pin:
            effective = max(0, cache_len - self.prefix_length)
        return (
            effective >= self.config.kv_budget
            and self.step_count > 0
            and (self.step_count % self.config.divide_length == 0)
        )

    def compress_cache(
        self,
        kv_cache: List[Tuple[mx.array, mx.array]],
    ) -> List[Tuple[mx.array, mx.array]]:
        """
        Compress KV cache by evicting low-importance tokens.

        kv_cache: list of (keys, values) per layer
                  keys: [batch, num_heads, seq_len, head_dim]

        Returns: compressed kv_cache
        """
        if not kv_cache:
            return kv_cache

        seq_len = kv_cache[0][0].shape[2]
        if seq_len <= self.config.kv_budget:
            return kv_cache

        # Score across all layers using norm-only scoring
        all_scores = []
        for layer_idx, (keys, _) in enumerate(kv_cache):
            # keys: [1, num_heads, seq_len, head_dim]
            layer_keys = keys[0]  # [num_heads, seq_len, head_dim]
            num_heads = layer_keys.shape[0]

            head_scores = []
            for h in range(num_heads):
                scores = score_keys_norm(layer_keys[h])  # [seq_len]
                head_scores.append(scores)

            stacked = mx.stack(head_scores, axis=0)  # [num_heads, seq_len]
            all_scores.append(stacked.mean(axis=0))  # [seq_len]

        # Aggregate across layers
        score_matrix = mx.stack(all_scores, axis=0)  # [num_layers, seq_len]
        global_scores = score_matrix.mean(axis=0)  # [seq_len]

        # Always keep prefill tokens
        prefix = self.prefix_length if self.config.prefill_pin else 0
        decode_scores = global_scores[prefix:]
        decode_len = decode_scores.shape[0]
        decode_budget = max(0, self.config.kv_budget - prefix)

        if decode_len <= decode_budget:
            return kv_cache

        # Top-k selection on decode tokens
        keep_k = min(decode_budget, decode_len)
        top_indices = mx.argsort(-decode_scores)[:keep_k]
        top_indices_sorted = mx.sort(top_indices)
        decode_keep_abs = top_indices_sorted + prefix

        # Combine: all prefill + selected decode
        prefill_indices = mx.arange(prefix)
        keep_indices = mx.concatenate([prefill_indices, decode_keep_abs])

        # Apply compression to each layer
        new_cache = []
        for keys, values in kv_cache:
            k_new = keys[:, :, keep_indices, :]
            v_new = values[:, :, keep_indices, :]
            new_cache.append((k_new, v_new))

        # Update position tracking
        keep_list = keep_indices.tolist()
        self.cache_positions = [self.cache_positions[i] for i in keep_list]
        self.compress_count += 1

        new_len = new_cache[0][0].shape[2]
        print(f"[TriAttention] Compressed: {seq_len} → {new_len} tokens (compression #{self.compress_count})")

        return new_cache

    def step(
        self,
        kv_cache: List[Tuple[mx.array, mx.array]],
        is_prefill: bool = False,
    ) -> List[Tuple[mx.array, mx.array]]:
        """
        Call after each decode step to manage KV compression.

        Args:
            kv_cache: Current KV cache from model
            is_prefill: True for first (prompt) forward pass

        Returns:
            (possibly compressed) KV cache
        """
        if is_prefill:
            seq_len = kv_cache[0][0].shape[2] if kv_cache else 0
            self.reset()
            self.cache_positions = list(range(seq_len))
            self.absolute_position = seq_len
            self.prefix_length = seq_len
            return kv_cache

        self.absolute_position += 1
        self.cache_positions.append(self.absolute_position - 1)
        self.step_count += 1

        cache_len = kv_cache[0][0].shape[2] if kv_cache else 0
        if self.should_compress(cache_len):
            kv_cache = self.compress_cache(kv_cache)

        return kv_cache
