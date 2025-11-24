"""
Attention Visualization Tool for Algorithm Distillation

This script extracts and visualizes transformer attention patterns to help
design compression strategies for the history context.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
from pathlib import Path

from dataset import ADDataset
from model import MODEL
from utils import get_config, get_data_loader
from einops import rearrange


class AttentionExtractor:
    """
    Extracts attention weights from transformer layers.
    Modified AD model to capture attention patterns.
    """
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention weights."""
        def get_attention_hook(name):
            def hook(module, input, output):
                # For TransformerEncoderLayer, we need to access the self-attention
                # The attention weights are computed in the MultiheadAttention module
                pass
            return hook
        
        # Register hooks on transformer encoder layers
        for i, layer in enumerate(self.model.transformer_encoder.layers):
            layer.register_forward_hook(get_attention_hook(f'layer_{i}'))
    
    def extract_attention_from_layer(self, layer_idx, x, attn_mask=None):
        """
        Manually extract attention weights from a specific layer.
        """
        # Get the specific layer
        layer = self.model.transformer_encoder.layers[layer_idx]
        
        # Store original forward to restore later
        original_self_attn_forward = layer.self_attn.forward
        
        attention_weights = []
        
        def custom_forward(query, key, value, key_padding_mask=None, 
                          need_weights=True, attn_mask=None, average_attn_weights=True,
                          is_causal=False):
            # Call original forward with need_weights=True
            output, attn_weights = original_self_attn_forward(
                query, key, value, 
                key_padding_mask=key_padding_mask,
                need_weights=True,  # Force returning attention weights
                attn_mask=attn_mask,
                average_attn_weights=False,  # Get per-head attention
                is_causal=is_causal
            )
            attention_weights.append(attn_weights.detach())
            return output, attn_weights
        
        # Temporarily replace forward method
        layer.self_attn.forward = custom_forward
        
        # Forward pass through this layer
        with torch.no_grad():
            _ = layer(x, src_mask=attn_mask)
        
        # Restore original forward
        layer.self_attn.forward = original_self_attn_forward
        
        return attention_weights[0] if attention_weights else None


def visualize_attention_heatmap(attention_weights, title="Attention Heatmap", 
                                save_path=None, figsize=(12, 10)):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: (num_heads, seq_len, seq_len) or (seq_len, seq_len)
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    if attention_weights.dim() == 3:
        # Average across heads or show each head
        num_heads = attention_weights.shape[0]
        
        fig, axes = plt.subplots(2, (num_heads + 1) // 2, figsize=figsize)
        axes = axes.flatten() if num_heads > 1 else [axes]
        
        for i in range(num_heads):
            ax = axes[i]
            sns.heatmap(attention_weights[i].cpu().numpy(), 
                       cmap='Reds', ax=ax, cbar=True,
                       xticklabels=False, yticklabels=False)
            ax.set_title(f'Head {i+1}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
        
        # Remove extra subplots
        for i in range(num_heads, len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()
    else:
        # Single attention matrix
        plt.figure(figsize=figsize)
        sns.heatmap(attention_weights.cpu().numpy(), 
                   cmap='Reds', cbar=True)
        plt.title(title)
        plt.xlabel('Key Position (History + Query)')
        plt.ylabel('Query Position (History + Query)')
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")
    
    plt.close()


def visualize_attention_lines(attention_weights, title="Attention Patterns",
                              save_path=None, figsize=(14, 8)):
    """
    Visualize attention weights as line plots showing which positions
    each query attends to.
    
    Args:
        attention_weights: (seq_len, seq_len) attention matrix
        title: Plot title
        save_path: Path to save figure
    """
    attn_np = attention_weights.cpu().numpy()
    seq_len = attn_np.shape[0]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Left plot: Show attention from query token to all history
    ax1 = axes[0]
    query_idx = seq_len - 1  # Last token (query state)
    ax1.plot(attn_np[query_idx, :], marker='o', linewidth=2, markersize=4)
    ax1.set_xlabel('Context Position', fontsize=12)
    ax1.set_ylabel('Attention Weight', fontsize=12)
    ax1.set_title('Query Token Attention to History', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=seq_len-1, color='r', linestyle='--', 
                label='Query Position', alpha=0.7)
    ax1.legend()
    
    # Right plot: Average attention across all queries to each position
    ax2 = axes[1]
    avg_attention = attn_np.mean(axis=0)
    ax2.bar(range(seq_len), avg_attention, alpha=0.7, color='steelblue')
    ax2.set_xlabel('Context Position', fontsize=12)
    ax2.set_ylabel('Average Attention Weight', fontsize=12)
    ax2.set_title('Average Attention to Each Position', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axvline(x=seq_len-1, color='r', linestyle='--', 
                label='Query Position', alpha=0.7)
    ax2.legend()
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention line plot to {save_path}")
    
    plt.close()


def analyze_attention_statistics(attention_weights, n_context_tokens):
    """
    Compute statistics about attention patterns.
    
    Args:
        attention_weights: (num_heads, seq_len, seq_len) or (seq_len, seq_len)
        n_context_tokens: Number of context tokens (excluding query)
    
    Returns:
        Dictionary with attention statistics
    """
    if attention_weights.dim() == 3:
        # Average across heads
        attn = attention_weights.mean(dim=0)
    else:
        attn = attention_weights
    
    attn_np = attn.cpu().numpy()
    seq_len = attn_np.shape[0]
    
    # Query token is the last one
    query_idx = seq_len - 1
    query_attention = attn_np[query_idx, :]
    
    stats = {
        'attention_to_query': query_attention[query_idx],
        'attention_to_context': query_attention[:n_context_tokens],
        'mean_context_attention': query_attention[:n_context_tokens].mean(),
        'std_context_attention': query_attention[:n_context_tokens].std(),
        'max_context_attention': query_attention[:n_context_tokens].max(),
        'max_context_position': query_attention[:n_context_tokens].argmax(),
        'attention_decay': [],  # Will measure if attention decays with distance
    }
    
    # Measure attention decay (recent vs distant history)
    if n_context_tokens > 1:
        quarter_size = max(1, n_context_tokens // 4)
        stats['recent_attention'] = query_attention[n_context_tokens-quarter_size:n_context_tokens].mean()
        stats['distant_attention'] = query_attention[:quarter_size].mean()
        stats['recent_to_distant_ratio'] = stats['recent_attention'] / (stats['distant_attention'] + 1e-10)
    
    # Entropy of attention distribution (higher = more uniform)
    attn_dist = query_attention[:n_context_tokens]
    entropy = -(attn_dist * np.log(attn_dist + 1e-10)).sum()
    stats['attention_entropy'] = entropy
    
    return stats


def visualize_attention_summary(all_attention_stats, save_path=None):
    """
    Visualize summary statistics across multiple samples.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract statistics
    mean_attn = [s['mean_context_attention'] for s in all_attention_stats]
    std_attn = [s['std_context_attention'] for s in all_attention_stats]
    entropy = [s['attention_entropy'] for s in all_attention_stats]
    ratios = [s.get('recent_to_distant_ratio', 0) for s in all_attention_stats]
    
    # Plot 1: Mean attention distribution
    ax1 = axes[0, 0]
    ax1.hist(mean_attn, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Mean Context Attention', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Mean Attention to Context', fontsize=13)
    ax1.axvline(np.mean(mean_attn), color='r', linestyle='--', 
                label=f'Mean: {np.mean(mean_attn):.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Attention std deviation
    ax2 = axes[0, 1]
    ax2.hist(std_attn, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax2.set_xlabel('Std Dev of Context Attention', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Attention Concentration (Std Dev)', fontsize=13)
    ax2.axvline(np.mean(std_attn), color='r', linestyle='--',
                label=f'Mean: {np.mean(std_attn):.4f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Attention entropy
    ax3 = axes[1, 0]
    ax3.hist(entropy, bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
    ax3.set_xlabel('Attention Entropy', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Attention Entropy (Uniformity)', fontsize=13)
    ax3.axvline(np.mean(entropy), color='r', linestyle='--',
                label=f'Mean: {np.mean(entropy):.4f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Recent vs distant attention ratio
    ax4 = axes[1, 1]
    ratios_filtered = [r for r in ratios if r > 0]
    if ratios_filtered:
        ax4.hist(ratios_filtered, bins=30, alpha=0.7, color='mediumpurple', edgecolor='black')
        ax4.set_xlabel('Recent/Distant Attention Ratio', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_title('Recent vs Distant Context Attention', fontsize=13)
        ax4.axvline(np.mean(ratios_filtered), color='r', linestyle='--',
                    label=f'Mean: {np.mean(ratios_filtered):.2f}')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Attention Pattern Statistics Summary', fontsize=16, y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved attention summary to {save_path}")
    
    plt.close()


def extract_and_visualize_attention(config, model, dataloader, num_samples=5, 
                                   layer_idx=0, save_dir='./figs/attention', 
                                   trajectory_idx=None):
    """
    Main function to extract and visualize attention patterns.
    
    Args:
        config: Configuration dict
        model: Trained AD model
        dataloader: Data loader
        num_samples: Number of timesteps to visualize from the trajectory
        layer_idx: Which transformer layer to analyze
        save_dir: Directory to save figures
        trajectory_idx: If provided, extract multiple timesteps from this single trajectory
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    extractor = AttentionExtractor(model)
    all_stats = []
    
    if trajectory_idx is not None:
        print(f"\nExtracting attention patterns from layer {layer_idx}...")
        print(f"Analyzing trajectory {trajectory_idx} at {num_samples} different timesteps")
        print("This shows how attention evolves during exploration and exploitation\n")
    else:
        print(f"\nExtracting attention patterns from layer {layer_idx}...")
        print(f"Number of samples to analyze: {num_samples}")
    
    with torch.no_grad():
        for sample_idx, batch in enumerate(dataloader):
            if trajectory_idx is None:
                if sample_idx >= num_samples:
                    break
            
            # If trajectory_idx is specified, only process that trajectory at different timesteps
            if trajectory_idx is not None:
                # Skip if this is not the target trajectory
                if sample_idx // (dataloader.dataset.__len__() // len(dataloader.dataset.states)) != trajectory_idx:
                    continue
                
                # Only process num_samples timesteps from this trajectory
                timestep_in_traj = sample_idx % (dataloader.dataset.__len__() // len(dataloader.dataset.states))
                if timestep_in_traj >= num_samples:
                    break
            
            # Prepare input exactly as in forward pass
            query_states = batch['query_states'].to(model.device)
            states = batch['states'].to(model.device)
            actions = batch['actions'].to(model.device)
            next_states = batch['next_states'].to(model.device)
            rewards = batch['rewards'].to(model.device)
            rewards = rearrange(rewards, 'b n -> b n 1')
            
            # Build transformer input
            from einops import pack
            from env import map_dark_states
            query_states_embed = model.embed_query_state(
                map_dark_states(query_states, config['grid_size']).to(torch.long)
            )
            query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')
            
            context, _ = pack([states, actions, rewards, next_states], 'b n *')
            context_embed = model.embed_context(context)
            context_embed, _ = pack([context_embed, query_states_embed], 'b * d')
            
            # Apply positional embedding
            x = model._apply_positional_embedding(context_embed)
            
            # Get causal mask
            seq_len = x.size(1)
            attn_mask = model._get_causal_mask(seq_len)
            
            # Forward through previous layers
            for i in range(layer_idx):
                layer = model.transformer_encoder.layers[i]
                x = layer(x, src_mask=attn_mask)
            
            # Extract attention from target layer
            attention = extractor.extract_attention_from_layer(layer_idx, x, attn_mask)
            
            if attention is None:
                print(f"Warning: Could not extract attention for sample {sample_idx}")
                continue
            
            # attention shape: (batch, num_heads, seq_len, seq_len)
            # Take first batch item
            attention_sample = attention[0]  # (num_heads, seq_len, seq_len)
            
            # Average across heads for easier visualization
            attention_avg = attention_sample.mean(dim=0)  # (seq_len, seq_len)
            
            n_context = seq_len - 1  # All tokens except the query
            
            # Compute statistics
            stats = analyze_attention_statistics(attention_sample, n_context)
            all_stats.append(stats)
            
            if trajectory_idx is not None:
                timestep = sample_idx % (dataloader.dataset.__len__() // len(dataloader.dataset.states))
                phase = "Early Exploration" if timestep < num_samples // 3 else ("Mid Exploration/Exploitation" if timestep < 2 * num_samples // 3 else "Late Exploitation")
                print(f"\nTimestep {timestep} ({phase}):")
            else:
                print(f"\nSample {sample_idx + 1}:")
            
            print(f"  Sequence length: {seq_len}")
            print(f"  Context tokens: {n_context}")
            print(f"  Mean context attention: {stats['mean_context_attention']:.4f}")
            print(f"  Attention std dev: {stats['std_context_attention']:.4f}")
            print(f"  Max attention position: {stats['max_context_position']}")
            if 'recent_to_distant_ratio' in stats:
                print(f"  Recent/Distant ratio: {stats['recent_to_distant_ratio']:.2f}")
            print(f"  Attention entropy: {stats['attention_entropy']:.4f}")
            
            # Visualize individual sample
            if trajectory_idx is not None:
                timestep = sample_idx % (dataloader.dataset.__len__() // len(dataloader.dataset.states))
                save_path_heatmap = os.path.join(save_dir, f'attention_heatmap_timestep{timestep}_layer{layer_idx}.png')
                sample_title_heatmap = f'Attention Heatmap - Timestep {timestep} ({phase}), Layer {layer_idx}'
                save_path_lines = os.path.join(save_dir, f'attention_lines_timestep{timestep}_layer{layer_idx}.png')
                sample_title_lines = f'Attention Patterns - Timestep {timestep} ({phase}), Layer {layer_idx}'
            else:
                save_path_heatmap = os.path.join(save_dir, f'attention_heatmap_sample{sample_idx}_layer{layer_idx}.png')
                sample_title_heatmap = f'Attention Heatmap - Sample {sample_idx + 1}, Layer {layer_idx}'
                save_path_lines = os.path.join(save_dir, f'attention_lines_sample{sample_idx}_layer{layer_idx}.png')
                sample_title_lines = f'Attention Patterns - Sample {sample_idx + 1}, Layer {layer_idx}'
            
            visualize_attention_heatmap(
                attention_sample,
                title=sample_title_heatmap,
                save_path=save_path_heatmap,
                figsize=(14, 10)
            )
            
            visualize_attention_lines(
                attention_avg,
                title=sample_title_lines,
                save_path=save_path_lines,
                figsize=(14, 6)
            )
    
    # Visualize summary statistics
    if all_stats:
        save_path_summary = os.path.join(save_dir, f'attention_summary_layer{layer_idx}.png')
        visualize_attention_summary(all_stats, save_path=save_path_summary)
        
        # Print overall statistics
        print("\n" + "="*60)
        print("OVERALL ATTENTION STATISTICS")
        print("="*60)
        print(f"Analyzed {len(all_stats)} samples")
        print(f"\nMean context attention: {np.mean([s['mean_context_attention'] for s in all_stats]):.4f} ± "
              f"{np.std([s['mean_context_attention'] for s in all_stats]):.4f}")
        print(f"Attention concentration (std): {np.mean([s['std_context_attention'] for s in all_stats]):.4f} ± "
              f"{np.std([s['std_context_attention'] for s in all_stats]):.4f}")
        print(f"Attention entropy: {np.mean([s['attention_entropy'] for s in all_stats]):.4f} ± "
              f"{np.std([s['attention_entropy'] for s in all_stats]):.4f}")
        ratios = [s.get('recent_to_distant_ratio', 0) for s in all_stats if 'recent_to_distant_ratio' in s]
        if ratios:
            print(f"Recent/Distant ratio: {np.mean(ratios):.2f} ± {np.std(ratios):.2f}")
            if np.mean(ratios) > 2.0:
                print("\n💡 INSIGHT: Strong recency bias detected!")
                print("   Consider compressing or removing distant history.")
            elif np.mean(ratios) < 1.2:
                print("\n💡 INSIGHT: Attention is spread across history!")
                print("   Compression may need to preserve diverse context.")
        
        entropy_mean = np.mean([s['attention_entropy'] for s in all_stats])
        if entropy_mean > 2.0:
            print("\n💡 INSIGHT: High attention entropy (uniform distribution)!")
            print("   Model attends broadly to context. Be careful with compression.")
        else:
            print("\n💡 INSIGHT: Low attention entropy (focused attention)!")
            print("   Model has clear attention preferences. Good for compression.")


def main():
    """Main execution function."""
    print("="*60)
    print("ATTENTION VISUALIZATION FOR ALGORITHM DISTILLATION")
    print("="*60)
    
    # Load configuration
    config = get_config('./config/env/darkroom.yaml')
    config.update(get_config('./config/algorithm/ppo_darkroom.yaml'))
    config.update(get_config('./config/model/ad_dr.yaml'))
    
    # Find latest checkpoint
    log_dir = f"./runs/{config['model']}-{config['env']}-seed{config['env_split_seed']}"
    ckpt_paths = sorted(glob(os.path.join(log_dir, 'ckpt-*.pt')))
    
    if not ckpt_paths:
        print(f"Error: No checkpoint found in {log_dir}")
        print("Please train the model first using train.py")
        return
    
    ckpt_path = ckpt_paths[-1]
    print(f"\nLoading checkpoint: {ckpt_path}")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    print(f"Using device: {device}")
    
    model = MODEL[config['model']](config)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Transformer layers: {len(model.transformer_encoder.layers)}")
    print(f"Attention heads: {model.transformer_encoder.layers[0].self_attn.num_heads}")
    print(f"Context length: {config['n_transit']}")
    
    # Load test dataset
    print("\nLoading test dataset...")
    config['traj_dir'] = './datasets'
    test_dataset = ADDataset(config, config['traj_dir'], 'test', 1, config['train_source_timesteps'])
    test_dataloader = get_data_loader(test_dataset, batch_size=1, config=config, shuffle=False)
    
    print(f"Dataset loaded: {len(test_dataset)} samples")
    
    # Extract and visualize attention for each layer
    num_layers = len(model.transformer_encoder.layers)
    
    # Choose visualization mode
    trajectory_mode = True  # Set to True to visualize single trajectory over time
    
    if trajectory_mode:
        trajectory_idx = 0  # Which trajectory to analyze
        num_timesteps = min(15, config['train_source_timesteps'])  # Number of timesteps to visualize
        
        print(f"\n{'='*60}")
        print("TRAJECTORY-BASED VISUALIZATION MODE")
        print(f"{'='*60}")
        print(f"Analyzing trajectory {trajectory_idx} across {num_timesteps} timesteps")
        print(f"This shows attention evolution during exploration → exploitation")
        print(f"Will analyze {num_layers} transformer layers\n")
        
        for layer_idx in range(num_layers):
            print(f"\n{'='*60}")
            print(f"ANALYZING LAYER {layer_idx + 1}/{num_layers}")
            print(f"{'='*60}")
            
            extract_and_visualize_attention(
                config, model, test_dataloader,
                num_samples=num_timesteps,
                layer_idx=layer_idx,
                save_dir=f'./figs/attention/trajectory_{trajectory_idx}/layer_{layer_idx}',
                trajectory_idx=trajectory_idx
            )
    else:
        num_samples = min(5, len(test_dataset))  # Visualize up to 5 samples
        
        print(f"\nWill analyze {num_samples} samples across {num_layers} layers")
        print("This may take a few minutes...\n")
        
        for layer_idx in range(num_layers):
            print(f"\n{'='*60}")
            print(f"ANALYZING LAYER {layer_idx + 1}/{num_layers}")
            print(f"{'='*60}")
            
            extract_and_visualize_attention(
                config, model, test_dataloader,
                num_samples=num_samples,
                layer_idx=layer_idx,
                save_dir=f'./figs/attention/layer_{layer_idx}'
            )
    
    print("\n" + "="*60)
    print("ATTENTION VISUALIZATION COMPLETE!")
    print("="*60)
    
    if trajectory_mode:
        print(f"Figures saved to: ./figs/attention/trajectory_{trajectory_idx}/")
        print("\nUse these visualizations to understand temporal attention patterns:")
        print("  - Early timesteps: Exploration phase, sparse context")
        print("  - Mid timesteps: Mixed exploration/exploitation")
        print("  - Late timesteps: Exploitation phase, rich context")
        print("\nCompression insights:")
        print("  - If attention stable across time → Compression strategy can be fixed")
        print("  - If attention changes over time → May need adaptive compression")
    else:
        print(f"Figures saved to: ./figs/attention/")
        print("\nUse these visualizations to design your compression strategy:")
        print("  - High recency bias → Compress/remove distant history")
        print("  - Uniform attention → Preserve diverse context samples")
        print("  - Sparse attention → Keep only high-attention positions")
        print("  - Clustered patterns → Group similar contexts")


if __name__ == '__main__':
    main()
