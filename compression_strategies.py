"""
Example Compression Implementations for Algorithm Distillation

This file provides template implementations for different compression strategies
based on attention patterns discovered through visualization.

After running visualize_attention.py, choose the compression strategy that
matches your attention pattern and integrate it into your AD model.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class HistoryCompressor:
    """Base class for history compression strategies."""
    
    def __init__(self, compression_ratio: float = 0.5):
        """
        Args:
            compression_ratio: Fraction of history to keep (0.0 to 1.0)
        """
        self.compression_ratio = compression_ratio
    
    def compress(self, states, actions, rewards, next_states):
        """
        Compress history tensors.
        
        Args:
            states: (batch, seq_len, state_dim)
            actions: (batch, seq_len, action_dim)
            rewards: (batch, seq_len, 1)
            next_states: (batch, seq_len, state_dim)
        
        Returns:
            Compressed tensors with same structure but reduced seq_len
        """
        raise NotImplementedError


class RecencyCompressor(HistoryCompressor):
    """
    Compression based on recency bias.
    Use when: Recent/Distant attention ratio > 2.0
    
    Strategy: Keep only the most recent K timesteps.
    """
    
    def __init__(self, keep_recent: int = 10):
        """
        Args:
            keep_recent: Number of recent timesteps to keep
        """
        super().__init__()
        self.keep_recent = keep_recent
    
    def compress(self, states, actions, rewards, next_states):
        """Keep only recent K steps."""
        k = min(self.keep_recent, states.size(1))
        
        return (
            states[:, -k:],
            actions[:, -k:],
            rewards[:, -k:],
            next_states[:, -k:]
        )


class ExponentialDecayCompressor(HistoryCompressor):
    """
    Compression with exponential decay sampling.
    Use when: Strong recency bias but need some distant context.
    
    Strategy: Keep more recent samples, exponentially fewer distant samples.
    """
    
    def __init__(self, target_length: int = 10, decay_rate: float = 0.5):
        """
        Args:
            target_length: Target compressed sequence length
            decay_rate: Exponential decay rate (0.0-1.0, lower = stronger decay)
        """
        super().__init__()
        self.target_length = target_length
        self.decay_rate = decay_rate
    
    def compress(self, states, actions, rewards, next_states):
        """Sample with exponential decay bias toward recent."""
        batch_size, seq_len, _ = states.shape
        
        if seq_len <= self.target_length:
            return states, actions, rewards, next_states
        
        # Generate sampling probabilities with exponential decay
        positions = torch.arange(seq_len, device=states.device)
        # Higher weight for recent positions
        weights = torch.exp(-self.decay_rate * (seq_len - 1 - positions))
        weights = weights / weights.sum()
        
        # Sample indices based on weights
        indices = torch.multinomial(weights, self.target_length, replacement=False)
        indices, _ = torch.sort(indices)  # Keep chronological order
        
        return (
            states[:, indices],
            actions[:, indices],
            rewards[:, indices],
            next_states[:, indices]
        )


class UniformCompressor(HistoryCompressor):
    """
    Uniform sampling compression.
    Use when: Attention entropy > 2.0 (uniform attention)
    
    Strategy: Sample evenly spaced points to preserve diversity.
    """
    
    def __init__(self, target_length: int = 10):
        """
        Args:
            target_length: Target compressed sequence length
        """
        super().__init__()
        self.target_length = target_length
    
    def compress(self, states, actions, rewards, next_states):
        """Sample uniformly spaced history points."""
        batch_size, seq_len, _ = states.shape
        
        if seq_len <= self.target_length:
            return states, actions, rewards, next_states
        
        # Generate evenly spaced indices
        indices = torch.linspace(0, seq_len - 1, self.target_length, 
                                device=states.device).long()
        
        return (
            states[:, indices],
            actions[:, indices],
            rewards[:, indices],
            next_states[:, indices]
        )


class AttentionBasedCompressor(HistoryCompressor):
    """
    Attention-weighted compression.
    Use when: Sparse, focused attention patterns
    
    Strategy: Keep positions with highest attention scores.
    
    Note: Requires access to attention weights from previous forward pass.
    """
    
    def __init__(self, target_length: int = 10, temperature: float = 1.0):
        """
        Args:
            target_length: Number of positions to keep
            temperature: Softmax temperature for attention sampling
        """
        super().__init__()
        self.target_length = target_length
        self.temperature = temperature
        self.cached_attention = None
    
    def cache_attention(self, attention_weights):
        """
        Cache attention weights from current forward pass.
        
        Args:
            attention_weights: (batch, num_heads, seq_len, seq_len)
                              or (batch, seq_len, seq_len)
        """
        if attention_weights.dim() == 4:
            # Average across heads
            attention_weights = attention_weights.mean(dim=1)
        
        # Take attention from query token (last position) to all context
        self.cached_attention = attention_weights[:, -1, :-1]  # (batch, context_len)
    
    def compress(self, states, actions, rewards, next_states):
        """Keep top-K attended positions."""
        if self.cached_attention is None:
            # Fallback to uniform sampling if no attention cached
            print("Warning: No cached attention, using uniform sampling")
            return UniformCompressor(self.target_length).compress(
                states, actions, rewards, next_states
            )
        
        batch_size, seq_len, _ = states.shape
        
        if seq_len <= self.target_length:
            return states, actions, rewards, next_states
        
        # Get top-K attended positions for each batch item
        k = min(self.target_length, self.cached_attention.size(1))
        topk_values, topk_indices = torch.topk(self.cached_attention, k, dim=1)
        
        # Sort indices to maintain chronological order
        topk_indices, _ = torch.sort(topk_indices, dim=1)
        
        # Gather positions
        batch_indices = torch.arange(batch_size, device=states.device).unsqueeze(1)
        
        compressed_states = states[batch_indices, topk_indices]
        compressed_actions = actions[batch_indices, topk_indices]
        compressed_rewards = rewards[batch_indices, topk_indices]
        compressed_next_states = next_states[batch_indices, topk_indices]
        
        return compressed_states, compressed_actions, compressed_rewards, compressed_next_states


class ThresholdCompressor(HistoryCompressor):
    """
    Threshold-based compression.
    Use when: Very sparse attention (only few positions matter)
    
    Strategy: Keep only positions above attention threshold.
    """
    
    def __init__(self, threshold: float = 0.1, min_keep: int = 3, max_keep: int = 20):
        """
        Args:
            threshold: Minimum attention weight to keep position
            min_keep: Minimum positions to keep (fallback)
            max_keep: Maximum positions to keep
        """
        super().__init__()
        self.threshold = threshold
        self.min_keep = min_keep
        self.max_keep = max_keep
        self.cached_attention = None
    
    def cache_attention(self, attention_weights):
        """Cache attention weights (same as AttentionBasedCompressor)."""
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=1)
        self.cached_attention = attention_weights[:, -1, :-1]
    
    def compress(self, states, actions, rewards, next_states):
        """Keep positions above threshold."""
        if self.cached_attention is None:
            return RecencyCompressor(self.max_keep).compress(
                states, actions, rewards, next_states
            )
        
        batch_size, seq_len, _ = states.shape
        
        # Find positions above threshold
        mask = self.cached_attention >= self.threshold  # (batch, seq_len)
        
        # Ensure minimum and maximum kept
        kept_counts = mask.sum(dim=1)
        
        compressed_batch = []
        for b in range(batch_size):
            if kept_counts[b] < self.min_keep:
                # Keep top min_keep if below minimum
                k = min(self.min_keep, seq_len)
                indices = torch.topk(self.cached_attention[b], k).indices
            elif kept_counts[b] > self.max_keep:
                # Keep top max_keep if above maximum
                indices = torch.topk(self.cached_attention[b], self.max_keep).indices
            else:
                # Keep all above threshold
                indices = torch.where(mask[b])[0]
            
            indices, _ = torch.sort(indices)
            
            compressed_batch.append((
                states[b, indices],
                actions[b, indices],
                rewards[b, indices],
                next_states[b, indices]
            ))
        
        # Pad to same length for batching
        max_len = max(len(item[0]) for item in compressed_batch)
        
        def pad_tensor(tensors_list, max_len):
            padded = []
            for t in tensors_list:
                pad_size = max_len - len(t)
                if pad_size > 0:
                    padding = torch.zeros(pad_size, *t.shape[1:], 
                                        device=t.device, dtype=t.dtype)
                    t = torch.cat([t, padding], dim=0)
                padded.append(t)
            return torch.stack(padded)
        
        return (
            pad_tensor([item[0] for item in compressed_batch], max_len),
            pad_tensor([item[1] for item in compressed_batch], max_len),
            pad_tensor([item[2] for item in compressed_batch], max_len),
            pad_tensor([item[3] for item in compressed_batch], max_len)
        )


class AdaptiveCompressor(HistoryCompressor):
    """
    Adaptive compression that combines multiple strategies.
    Use when: Complex attention patterns
    
    Strategy: Automatically selects compression based on attention statistics.
    """
    
    def __init__(self, target_length: int = 10):
        super().__init__()
        self.target_length = target_length
        self.recency_compressor = RecencyCompressor(target_length)
        self.uniform_compressor = UniformCompressor(target_length)
        self.attention_compressor = AttentionBasedCompressor(target_length)
    
    def cache_attention(self, attention_weights):
        """Cache for attention-based compression."""
        self.attention_compressor.cache_attention(attention_weights)
    
    def _compute_attention_stats(self, attention_weights):
        """Compute statistics to determine which strategy to use."""
        if attention_weights.dim() == 4:
            attention_weights = attention_weights.mean(dim=1)
        
        query_attention = attention_weights[:, -1, :-1]  # (batch, context_len)
        
        # Compute recency bias
        context_len = query_attention.size(1)
        if context_len > 4:
            quarter = context_len // 4
            recent_attn = query_attention[:, -quarter:].mean()
            distant_attn = query_attention[:, :quarter].mean()
            recency_ratio = recent_attn / (distant_attn + 1e-10)
        else:
            recency_ratio = 1.0
        
        # Compute entropy
        entropy = -(query_attention * torch.log(query_attention + 1e-10)).sum(dim=1).mean()
        
        return recency_ratio, entropy
    
    def compress(self, states, actions, rewards, next_states):
        """Adaptively choose compression strategy."""
        if self.attention_compressor.cached_attention is not None:
            # Reconstruct full attention for stats
            attention = self.attention_compressor.cached_attention
            recency_ratio, entropy = self._compute_attention_stats(
                attention.unsqueeze(1).unsqueeze(1)  # Add head and query dims
            )
            
            # Decision logic
            if recency_ratio > 2.5:
                # Strong recency bias
                return self.recency_compressor.compress(
                    states, actions, rewards, next_states
                )
            elif entropy > 2.0:
                # Uniform attention
                return self.uniform_compressor.compress(
                    states, actions, rewards, next_states
                )
            else:
                # Use attention-based compression
                return self.attention_compressor.compress(
                    states, actions, rewards, next_states
                )
        else:
            # Default to recency-based
            return self.recency_compressor.compress(
                states, actions, rewards, next_states
            )


# ============================================================================
# Integration Example: How to modify AD model
# ============================================================================

def example_integration():
    """
    Example of how to integrate compression into your AD model.
    
    Modify model/ad.py by adding compression in the forward method.
    """
    
    # In ad.py, add at the top:
    # from compression_strategies import RecencyCompressor  # or other strategy
    
    # In AD.__init__, add:
    # self.compressor = RecencyCompressor(keep_recent=10)
    
    # In AD.forward, after building context_embed but before transformer:
    """
    # Original code:
    context, _ = pack([states, actions, rewards, next_states], 'b n *')
    context_embed = self.embed_context(context)
    context_embed, _ = pack([context_embed, query_states_embed], 'b * d')
    
    # Modified with compression:
    # Compress history before embedding
    states_compressed, actions_compressed, rewards_compressed, next_states_compressed = \
        self.compressor.compress(states, actions, rewards, next_states)
    
    context, _ = pack([states_compressed, actions_compressed, 
                      rewards_compressed, next_states_compressed], 'b n *')
    context_embed = self.embed_context(context)
    context_embed, _ = pack([context_embed, query_states_embed], 'b * d')
    """
    
    # For attention-based compression, cache attention after transformer:
    """
    transformer_output = self.transformer(context_embed, ...)
    
    # If using AttentionBasedCompressor, extract and cache attention
    # This requires modifying the transformer to return attention weights
    if hasattr(self, 'compressor') and hasattr(self.compressor, 'cache_attention'):
        # Extract attention from last layer (requires model modification)
        attention = self._extract_attention_weights()
        self.compressor.cache_attention(attention)
    """


# ============================================================================
# Testing and Validation
# ============================================================================

def test_compressor(compressor, seq_len=20, batch_size=4):
    """Test a compressor with dummy data."""
    print(f"\nTesting {compressor.__class__.__name__}...")
    
    # Create dummy data
    states = torch.randn(batch_size, seq_len, 2)
    actions = F.one_hot(torch.randint(0, 5, (batch_size, seq_len)), 5).float()
    rewards = torch.randn(batch_size, seq_len, 1)
    next_states = torch.randn(batch_size, seq_len, 2)
    
    # For attention-based, create dummy attention
    if hasattr(compressor, 'cache_attention'):
        attention = F.softmax(torch.randn(batch_size, 4, seq_len, seq_len), dim=-1)
        compressor.cache_attention(attention)
    
    # Compress
    s_comp, a_comp, r_comp, ns_comp = compressor.compress(
        states, actions, rewards, next_states
    )
    
    print(f"Original length: {seq_len}")
    print(f"Compressed length: {s_comp.size(1)}")
    print(f"Compression ratio: {s_comp.size(1) / seq_len:.2%}")
    print(f"Memory reduction: {(1 - s_comp.size(1) / seq_len):.2%}")


if __name__ == '__main__':
    print("="*60)
    print("COMPRESSION STRATEGIES TEST")
    print("="*60)
    
    # Test all compressors
    test_compressor(RecencyCompressor(keep_recent=10))
    test_compressor(ExponentialDecayCompressor(target_length=10))
    test_compressor(UniformCompressor(target_length=10))
    test_compressor(AttentionBasedCompressor(target_length=10))
    test_compressor(ThresholdCompressor(threshold=0.1))
    test_compressor(AdaptiveCompressor(target_length=10))
    
    print("\n" + "="*60)
    print("All compressors tested successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run visualize_attention.py to analyze your model")
    print("2. Choose appropriate compression strategy based on results")
    print("3. Integrate chosen compressor into model/ad.py")
    print("4. Retrain and evaluate performance")
