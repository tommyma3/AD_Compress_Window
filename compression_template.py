"""
Example implementation of attention-based history compression.
This shows how to integrate compression into the AD model.

This is a TEMPLATE - adjust based on your attention analysis results!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class AttentionBasedCompression(nn.Module):
    """
    Compresses history context based on attention patterns.
    
    Strategies implemented:
    1. Top-K selection: Keep positions with highest attention
    2. Attention-weighted pooling: Compress low-attention positions
    3. Fixed window: Keep recent + compress old
    """
    
    def __init__(self, compression_ratio=0.5, strategy='topk'):
        super().__init__()
        self.compression_ratio = compression_ratio
        self.strategy = strategy
    
    def forward(self, context_embeddings, attention_weights):
        """
        Compress context based on attention patterns.
        
        Args:
            context_embeddings: (batch, seq_len, emb_dim)
            attention_weights: (batch, num_heads, seq_len, seq_len)
        
        Returns:
            compressed_embeddings: (batch, compressed_len, emb_dim)
        """
        if self.strategy == 'topk':
            return self._compress_topk(context_embeddings, attention_weights)
        elif self.strategy == 'pooling':
            return self._compress_pooling(context_embeddings, attention_weights)
        elif self.strategy == 'fixed_window':
            return self._compress_fixed_window(context_embeddings, attention_weights)
        else:
            return context_embeddings
    
    def _compress_topk(self, embeddings, attention):
        """Keep top-K positions by attention weight."""
        batch_size, seq_len, emb_dim = embeddings.shape
        
        # Average attention from all queries to each key position
        # (batch, heads, seq, seq) -> (batch, seq)
        avg_attention = attention.mean(dim=1).mean(dim=1)
        
        # Determine how many positions to keep
        k = max(1, int(seq_len * self.compression_ratio))
        
        # Get top-k positions
        topk_values, topk_indices = torch.topk(avg_attention, k, dim=1)
        
        # Sort indices to maintain temporal order
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=1)
        
        # Select embeddings at top-k positions
        compressed = torch.gather(
            embeddings, 
            dim=1,
            index=topk_indices_sorted.unsqueeze(-1).expand(-1, -1, emb_dim)
        )
        
        return compressed
    
    def _compress_pooling(self, embeddings, attention):
        """Compress using attention-weighted pooling."""
        batch_size, seq_len, emb_dim = embeddings.shape
        
        # Compute average attention to each position
        avg_attention = attention.mean(dim=1).mean(dim=1)  # (batch, seq)
        
        # Determine compression groups
        target_len = max(1, int(seq_len * self.compression_ratio))
        window_size = seq_len // target_len
        
        compressed_parts = []
        for i in range(target_len):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, seq_len)
            
            if start_idx >= seq_len:
                break
            
            # Get embeddings and attention for this window
            window_emb = embeddings[:, start_idx:end_idx, :]  # (batch, window, emb)
            window_attn = avg_attention[:, start_idx:end_idx]  # (batch, window)
            
            # Normalize attention weights
            window_attn = F.softmax(window_attn, dim=1)
            
            # Weighted sum
            compressed_window = torch.sum(
                window_emb * window_attn.unsqueeze(-1), 
                dim=1, 
                keepdim=True
            )  # (batch, 1, emb)
            
            compressed_parts.append(compressed_window)
        
        return torch.cat(compressed_parts, dim=1)
    
    def _compress_fixed_window(self, embeddings, attention, recent_window=10):
        """Keep recent positions, compress older ones."""
        batch_size, seq_len, emb_dim = embeddings.shape
        
        if seq_len <= recent_window:
            return embeddings
        
        # Keep recent positions unchanged
        recent_part = embeddings[:, -recent_window:, :]
        
        # Compress older positions
        old_part = embeddings[:, :-recent_window, :]
        old_attention = attention[:, :, -recent_window:, :-recent_window].mean(dim=(1, 2))
        
        # Determine how many old positions to keep
        old_len = seq_len - recent_window
        target_old_len = max(1, int(old_len * 0.3))  # Keep 30% of old context
        
        # Top-k from old positions
        topk_values, topk_indices = torch.topk(old_attention, target_old_len, dim=1)
        topk_indices_sorted, _ = torch.sort(topk_indices, dim=1)
        
        compressed_old = torch.gather(
            old_part,
            dim=1,
            index=topk_indices_sorted.unsqueeze(-1).expand(-1, -1, emb_dim)
        )
        
        # Concatenate compressed old + recent
        return torch.cat([compressed_old, recent_part], dim=1)


class ADWithCompression(nn.Module):
    """
    Example: AD model with integrated compression.
    
    This is a TEMPLATE showing where to add compression.
    Integrate this into your actual model/ad.py.
    """
    
    def __init__(self, config):
        super().__init__()
        # ... (your existing AD initialization)
        
        # Add compression module
        self.use_compression = config.get('use_compression', False)
        self.compression_interval = config.get('compression_interval', 20)
        
        if self.use_compression:
            self.compression_module = AttentionBasedCompression(
                compression_ratio=config.get('compression_ratio', 0.5),
                strategy=config.get('compression_strategy', 'fixed_window')
            )
    
    def evaluate_in_context_with_compression(self, vec_env, eval_timesteps):
        """
        Modified evaluation with compression.
        
        Key changes:
        1. Capture attention weights during forward pass
        2. Apply compression at regular intervals
        3. Track sequence length for speedup measurement
        """
        outputs = {}
        outputs['reward_episode'] = []
        outputs['sequence_lengths'] = []  # Track compression effect
        
        reward_episode = torch.zeros(vec_env.num_envs)
        
        # ... (initialization code)
        
        for step in range(eval_timesteps):
            # Forward pass with attention capture
            seq_len = transformer_input.size(1)
            outputs['sequence_lengths'].append(seq_len)
            
            # Get attention weights (you need to modify transformer to return these)
            transformer_output, attention_weights = self.transformer_with_attention(
                transformer_input
            )
            
            # ... (get actions and step environment)
            
            # Compression at intervals
            if self.use_compression and (step + 1) % self.compression_interval == 0:
                transformer_input = self.compression_module(
                    transformer_input,
                    attention_weights
                )
                print(f"Step {step+1}: Compressed {seq_len} -> {transformer_input.size(1)}")
            
            # ... (continue with normal flow)
        
        return outputs


# Example configuration for compression
COMPRESSION_CONFIG = {
    # Enable compression
    'use_compression': True,
    
    # Compression interval (steps between compressions)
    'compression_interval': 20,
    
    # Compression ratio (0.5 = keep 50% of positions)
    'compression_ratio': 0.5,
    
    # Strategy: 'topk', 'pooling', or 'fixed_window'
    'compression_strategy': 'fixed_window',
}


def example_usage():
    """
    Example of how to use compression in your training/evaluation.
    """
    
    # 1. Add compression config to your existing config
    config = {
        # ... your existing config
        **COMPRESSION_CONFIG
    }
    
    # 2. Modify your model to support compression
    # model = ADWithCompression(config)
    
    # 3. During evaluation, track speedup
    import time
    
    # Without compression
    # start = time.time()
    # results_no_compression = model.evaluate_in_context(...)
    # time_no_compression = time.time() - start
    
    # With compression
    # start = time.time()
    # results_with_compression = model.evaluate_in_context_with_compression(...)
    # time_with_compression = time.time() - start
    
    # speedup = time_no_compression / time_with_compression
    # print(f"Speedup: {speedup:.2f}x")
    
    pass


if __name__ == '__main__':
    print("="*80)
    print("ATTENTION-BASED HISTORY COMPRESSION TEMPLATE")
    print("="*80)
    print()
    print("This file shows how to implement compression in your AD model.")
    print()
    print("Steps to integrate:")
    print("  1. Run attention analysis first (run_attention_analysis.py)")
    print("  2. Review the compression recommendations")
    print("  3. Choose a compression strategy based on your attention patterns")
    print("  4. Modify model/ad.py to integrate compression:")
    print("     - Add compression module to __init__")
    print("     - Modify transformer to return attention weights")
    print("     - Add compression logic to evaluate_in_context")
    print("  5. Test and measure speedup vs performance tradeoff")
    print()
    print("Example strategies:")
    print("  - Top-K: Keep positions with highest attention")
    print("  - Pooling: Attention-weighted merge of low-attention positions")
    print("  - Fixed Window: Keep recent, compress old (often works best)")
    print()
    print("="*80)
