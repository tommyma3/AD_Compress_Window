"""
Visualize attention patterns during evaluation to guide history compression.
"""
from datetime import datetime
from glob import glob
import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
seed = 0
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def extract_attention_weights(model, hook_handles):
    """Extract attention weights from transformer encoder layers."""
    attention_weights = []
    
    def attention_hook(module, input, output):
        # For TransformerEncoderLayer, we need to hook into the self_attn module
        pass
    
    # Hook into each transformer encoder layer's self-attention
    for layer in model.transformer_encoder.layers:
        def make_hook(layer_idx):
            def hook(module, input, output):
                # The self_attn module returns (attn_output, attn_weights)
                # But TransformerEncoderLayer doesn't return weights by default
                # We need to access them during forward pass
                pass
            return hook
        
        handle = layer.self_attn.register_forward_hook(make_hook(len(hook_handles)))
        hook_handles.append(handle)
    
    return attention_weights


def visualize_attention_heatmap(attention_weights, save_path, title="Attention Patterns"):
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: List of attention matrices (num_layers, num_heads, seq_len, seq_len)
        save_path: Path to save the visualization
        title: Title for the plot
    """
    num_layers = len(attention_weights)
    num_heads = attention_weights[0].shape[0]
    
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(4 * num_heads, 4 * num_layers))
    
    if num_layers == 1 and num_heads == 1:
        axes = np.array([[axes]])
    elif num_layers == 1:
        axes = axes.reshape(1, -1)
    elif num_heads == 1:
        axes = axes.reshape(-1, 1)
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            ax = axes[layer_idx, head_idx]
            attn = attention_weights[layer_idx][head_idx]
            
            sns.heatmap(attn, ax=ax, cmap='viridis', cbar=True, 
                       square=True, vmin=0, vmax=1)
            ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
    
    plt.suptitle(title, fontsize=16, y=1.002)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention heatmap to {save_path}")


def visualize_attention_aggregated(attention_history, save_path):
    """
    Visualize aggregated attention patterns over time.
    
    Args:
        attention_history: List of attention weights over timesteps
        save_path: Path to save the visualization
    """
    # Average over layers and heads to get overall attention patterns
    aggregated = []
    for attn_weights in attention_history:
        # Average over layers and heads: (layers, heads, seq, seq) -> (seq, seq)
        avg_attn = np.mean([np.mean(layer_attn, axis=0) for layer_attn in attn_weights], axis=0)
        aggregated.append(avg_attn)
    
    # Create visualization
    fig, axes = plt.subplots(1, min(len(aggregated), 5), figsize=(20, 4))
    
    if len(aggregated) == 1:
        axes = [axes]
    
    for idx, (ax, attn) in enumerate(zip(axes, aggregated[:5])):
        sns.heatmap(attn, ax=ax, cmap='viridis', cbar=True, square=True, vmin=0, vmax=1)
        ax.set_title(f'Step {idx+1}')
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
    
    plt.suptitle('Attention Patterns Over Time (Averaged)', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved aggregated attention visualization to {save_path}")


def visualize_attention_to_positions(attention_history, save_path):
    """
    Visualize how attention changes to different positions over time.
    Shows which history positions are most attended to.
    
    Args:
        attention_history: List of attention weights over timesteps
        save_path: Path to save the visualization
    """
    attention_to_positions = []
    
    for attn_weights in attention_history:
        # Average over layers and heads
        avg_attn = np.mean([np.mean(layer_attn, axis=0) for layer_attn in attn_weights], axis=0)
        # Get attention from the last query position to all key positions
        if avg_attn.shape[0] > 0:
            attention_to_positions.append(avg_attn[-1, :])
    
    if len(attention_to_positions) == 0:
        print("No attention data to visualize")
        return
    
    # Convert to array: (timesteps, positions)
    attention_matrix = np.array(attention_to_positions)
    
    plt.figure(figsize=(12, 6))
    
    # Plot heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(attention_matrix.T, cmap='viridis', cbar=True, 
               xticklabels=range(1, len(attention_to_positions) + 1))
    plt.xlabel('Evaluation Step')
    plt.ylabel('Context Position')
    plt.title('Attention to Context Positions Over Time')
    
    # Plot line chart for specific positions
    plt.subplot(1, 2, 2)
    max_pos = min(attention_matrix.shape[1], 20)  # Show up to 20 positions
    for pos in range(0, max_pos, max(1, max_pos // 5)):
        plt.plot(attention_matrix[:, pos], label=f'Pos {pos}', alpha=0.7)
    plt.xlabel('Evaluation Step')
    plt.ylabel('Attention Weight')
    plt.title('Attention Weight Trends')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved attention-to-positions visualization to {save_path}")


def visualize_compression_potential(attention_history, save_path, window_size=10):
    """
    Analyze compression potential by identifying low-attention positions.
    
    Args:
        attention_history: List of attention weights over timesteps
        save_path: Path to save the visualization
        window_size: Window size for computing compression potential
    """
    # Compute average attention to each position
    all_positions_attention = []
    
    for attn_weights in attention_history:
        avg_attn = np.mean([np.mean(layer_attn, axis=0) for layer_attn in attn_weights], axis=0)
        # Average attention TO each position (average over query dimension)
        attention_to_pos = avg_attn.mean(axis=0)
        all_positions_attention.append(attention_to_pos)
    
    if len(all_positions_attention) == 0:
        print("No attention data to analyze")
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Average attention to each position across all timesteps
    avg_attention_per_pos = np.mean(all_positions_attention, axis=0)
    axes[0].bar(range(len(avg_attention_per_pos)), avg_attention_per_pos)
    axes[0].axhline(y=1.0/len(avg_attention_per_pos), color='r', linestyle='--', 
                    label='Uniform attention')
    axes[0].set_xlabel('Context Position')
    axes[0].set_ylabel('Average Attention Weight')
    axes[0].set_title('Average Attention to Each Position (Compression Candidates)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Cumulative attention (identify most important positions)
    sorted_indices = np.argsort(avg_attention_per_pos)[::-1]
    cumsum = np.cumsum(avg_attention_per_pos[sorted_indices])
    axes[1].plot(range(len(cumsum)), cumsum, marker='o')
    axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% attention')
    axes[1].axhline(y=0.9, color='orange', linestyle='--', label='90% attention')
    axes[1].set_xlabel('Number of Top Positions Kept')
    axes[1].set_ylabel('Cumulative Attention')
    axes[1].set_title('Cumulative Attention Distribution (Compression Strategy)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved compression potential analysis to {save_path}")
    
    # Print statistics
    positions_for_80 = np.where(cumsum >= 0.8)[0][0] + 1 if np.any(cumsum >= 0.8) else len(cumsum)
    positions_for_90 = np.where(cumsum >= 0.9)[0][0] + 1 if np.any(cumsum >= 0.9) else len(cumsum)
    print(f"\nCompression Analysis:")
    print(f"  Total context positions: {len(avg_attention_per_pos)}")
    print(f"  Positions for 80% attention: {positions_for_80} ({100*positions_for_80/len(avg_attention_per_pos):.1f}%)")
    print(f"  Positions for 90% attention: {positions_for_90} ({100*positions_for_90/len(avg_attention_per_pos):.1f}%)")


if __name__ == '__main__':
    ckpt_dir = './runs/AD-darkroom-seed0'
    output_dir = './figs/attention_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    ckpt_paths = sorted(glob(path.join(ckpt_dir, 'ckpt-*.pt')))

    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path)
        print(f'Checkpoint loaded from {ckpt_path}')
        config = ckpt['config']
    else:
        raise ValueError('No checkpoint found.')
    
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Enable attention weight return in transformer
    # We'll need to modify the model to capture attention
    print("\nNote: This script requires modifying the model to capture attention weights.")
    print("See the modified version of AD model below.\n")

    env_name = config['env']
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)

    print("Evaluation goals: ", test_env_args)

    if env_name == 'darkroom':
        # Use single environment for detailed attention analysis
        envs = SubprocVecEnv([make_env(config, goal=test_env_args[0])])
    else:
        raise NotImplementedError(f'Environment not supported')
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = datetime.now()
    print(f'Starting attention analysis at {start_time}')

    # Storage for attention patterns
    attention_history = []
    
    # Note: The actual implementation requires modifying the AD model
    # to return attention weights. See evaluate_attention.py for the implementation.
    
    print("\n" + "="*80)
    print("To capture attention weights, you need to:")
    print("1. Use evaluate_with_attention.py (the modified version)")
    print("2. This will capture attention patterns during evaluation")
    print("3. Then run visualize_attention.py to create visualizations")
    print("="*80)
    
    envs.close()
