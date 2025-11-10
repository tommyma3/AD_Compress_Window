"""
Analyze and visualize captured attention patterns to guide compression strategy.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")


def load_attention_data(data_path, attention_path):
    """Load captured attention data."""
    import pickle
    
    # Load rewards from npz file
    data = np.load(data_path, allow_pickle=True)
    rewards = data['rewards']
    
    # Load attention history from pickle file (handles variable-length sequences)
    with open(attention_path, 'rb') as f:
        attention_history = pickle.load(f)
    
    return attention_history, rewards


def visualize_attention_heatmaps(attention_history, output_dir, num_samples=5):
    """
    Visualize attention heatmaps at different timesteps.
    
    Args:
        attention_history: List of attention weights over timesteps
        output_dir: Directory to save visualizations
        num_samples: Number of timesteps to visualize
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select evenly spaced timesteps
    total_steps = len(attention_history)
    sample_indices = np.linspace(0, total_steps-1, num_samples, dtype=int)
    
    for sample_idx, step_idx in enumerate(sample_indices):
        attention_layers = attention_history[step_idx]
        num_layers = len(attention_layers)
        
        if num_layers == 0:
            continue
        
        # Get first sample in batch (batch_size, num_heads, seq, seq)
        num_heads = attention_layers[0].shape[1]
        
        fig, axes = plt.subplots(num_layers, num_heads, 
                                figsize=(4 * num_heads, 4 * num_layers),
                                squeeze=False)
        
        for layer_idx in range(num_layers):
            attn = attention_layers[layer_idx][0]  # First batch element
            
            for head_idx in range(num_heads):
                ax = axes[layer_idx, head_idx]
                
                # Plot attention heatmap
                im = ax.imshow(attn[head_idx], cmap='viridis', aspect='auto', 
                              vmin=0, vmax=1)
                ax.set_title(f'Layer {layer_idx+1}, Head {head_idx+1}')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle(f'Attention Patterns at Step {step_idx+1}/{total_steps}', 
                    fontsize=16, y=1.002)
        plt.tight_layout()
        
        save_path = os.path.join(output_dir, f'attention_heatmap_step{step_idx+1:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def visualize_attention_to_positions(attention_history, output_dir):
    """
    Visualize how attention to different context positions evolves over time.
    This is crucial for compression strategy.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract attention from last query position to all key positions
    attention_to_positions = []
    sequence_lengths = []
    
    for step_idx, attention_layers in enumerate(attention_history):
        if len(attention_layers) == 0:
            continue
        
        # Average over layers and heads
        all_layers = []
        for layer_attn in attention_layers:
            # layer_attn shape: (batch, heads, seq, seq)
            avg_heads = layer_attn[0].mean(axis=0)  # Average over heads
            all_layers.append(avg_heads)
        
        avg_attn = np.mean(all_layers, axis=0)  # Average over layers
        
        # Get attention from last position (current query) to all positions
        seq_len = avg_attn.shape[0]
        sequence_lengths.append(seq_len)
        
        if seq_len > 0:
            attention_to_positions.append(avg_attn[-1, :])
    
    if len(attention_to_positions) == 0:
        print("No attention data to visualize")
        return
    
    # Pad sequences to same length for visualization
    max_len = max(len(attn) for attn in attention_to_positions)
    attention_matrix = np.zeros((len(attention_to_positions), max_len))
    
    for i, attn in enumerate(attention_to_positions):
        attention_matrix[i, :len(attn)] = attn
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. Full attention heatmap over time
    ax1 = fig.add_subplot(gs[0, :])
    im = ax1.imshow(attention_matrix.T, aspect='auto', cmap='viridis', 
                   interpolation='nearest')
    ax1.set_xlabel('Evaluation Step', fontsize=12)
    ax1.set_ylabel('Context Position', fontsize=12)
    ax1.set_title('Attention from Current Query to All Context Positions', fontsize=14)
    plt.colorbar(im, ax=ax1, label='Attention Weight')
    
    # 2. Attention to recent vs distant positions
    ax2 = fig.add_subplot(gs[1, 0])
    
    # For each timestep, compute attention to recent (last 5) vs distant positions
    recent_attention = []
    distant_attention = []
    
    for i, attn in enumerate(attention_to_positions):
        if len(attn) > 20:
            recent_attention.append(attn[-20:].sum())
            distant_attention.append(attn[:-20].sum())
        else:
            recent_attention.append(attn.sum())
            distant_attention.append(0)
    
    steps = range(len(recent_attention))
    ax2.plot(steps, recent_attention, label='Recent (last 20)', linewidth=2)
    ax2.plot(steps, distant_attention, label='Distant (rest)', linewidth=2)
    ax2.set_xlabel('Evaluation Step', fontsize=12)
    ax2.set_ylabel('Cumulative Attention', fontsize=12)
    ax2.set_title('Attention: Recent vs Distant Context', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Sequence length growth
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(sequence_lengths, linewidth=2, color='darkblue')
    ax3.set_xlabel('Evaluation Step', fontsize=12)
    ax3.set_ylabel('Sequence Length', fontsize=12)
    ax3.set_title('Context Length Growth Over Time', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # 4. Average attention by relative position
    ax4 = fig.add_subplot(gs[2, 0])
    
    # Compute average attention to positions relative to current query
    relative_attention = {}
    for attn in attention_to_positions:
        seq_len = len(attn)
        for pos in range(seq_len):
            relative_pos = pos - (seq_len - 1)  # Relative to last position
            if relative_pos not in relative_attention:
                relative_attention[relative_pos] = []
            relative_attention[relative_pos].append(attn[pos])
    
    rel_positions = sorted(relative_attention.keys())
    avg_attention = [np.mean(relative_attention[pos]) for pos in rel_positions]
    std_attention = [np.std(relative_attention[pos]) for pos in rel_positions]
    
    ax4.bar(rel_positions, avg_attention, alpha=0.7, color='steelblue')
    ax4.errorbar(rel_positions, avg_attention, yerr=std_attention, 
                fmt='none', ecolor='black', alpha=0.3, capsize=3)
    ax4.set_xlabel('Relative Position (0 = current)', fontsize=12)
    ax4.set_ylabel('Average Attention Weight', fontsize=12)
    ax4.set_title('Attention by Relative Position', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 5. Compression potential
    ax5 = fig.add_subplot(gs[2, 1])
    
    # Average attention to each absolute position across all timesteps
    position_attention = {}
    for attn in attention_to_positions:
        for pos, weight in enumerate(attn):
            if pos not in position_attention:
                position_attention[pos] = []
            position_attention[pos].append(weight)
    
    avg_attn_per_pos = {pos: np.mean(weights) for pos, weights in position_attention.items()}
    
    if len(avg_attn_per_pos) > 0:
        positions = sorted(avg_attn_per_pos.keys())
        attentions = [avg_attn_per_pos[pos] for pos in positions]
        
        # Sort by attention weight for cumulative plot
        sorted_indices = np.argsort(attentions)[::-1]
        sorted_attentions = np.array(attentions)[sorted_indices]
        cumsum = np.cumsum(sorted_attentions)
        
        ax5.plot(range(len(cumsum)), cumsum, linewidth=2, marker='o', markersize=4)
        ax5.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80% attention')
        ax5.axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90% attention')
        ax5.set_xlabel('Number of Top Positions Kept', fontsize=12)
        ax5.set_ylabel('Cumulative Attention', fontsize=12)
        ax5.set_title('Compression Potential Analysis', fontsize=14)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Print compression statistics
        if len(cumsum) > 0:
            pos_80 = np.where(cumsum >= 0.8)[0]
            pos_90 = np.where(cumsum >= 0.9)[0]
            
            if len(pos_80) > 0:
                n_80 = pos_80[0] + 1
                print(f"\nCompression Analysis:")
                print(f"  Total positions analyzed: {len(cumsum)}")
                print(f"  Positions for 80% attention: {n_80} ({100*n_80/len(cumsum):.1f}%)")
                
            if len(pos_90) > 0:
                n_90 = pos_90[0] + 1
                print(f"  Positions for 90% attention: {n_90} ({100*n_90/len(cumsum):.1f}%)")
    
    save_path = os.path.join(output_dir, 'attention_analysis_comprehensive.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comprehensive analysis: {save_path}")


def analyze_layer_head_patterns(attention_history, output_dir):
    """
    Analyze attention patterns across different layers and heads.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if len(attention_history) == 0 or len(attention_history[0]) == 0:
        print("No attention data to analyze")
        return
    
    num_layers = len(attention_history[0])
    num_heads = attention_history[0][0].shape[1]
    
    # Compute average attention entropy for each layer and head
    layer_head_entropy = np.zeros((num_layers, num_heads))
    layer_head_max_attention = np.zeros((num_layers, num_heads))
    
    for attention_layers in attention_history:
        for layer_idx, layer_attn in enumerate(attention_layers):
            attn = layer_attn[0]  # First batch element
            
            for head_idx in range(num_heads):
                head_attn = attn[head_idx]
                
                # Compute entropy (measure of attention spread)
                # Average over query positions
                for query_attn in head_attn:
                    query_attn = query_attn + 1e-10  # Avoid log(0)
                    entropy = -(query_attn * np.log(query_attn)).sum()
                    layer_head_entropy[layer_idx, head_idx] += entropy
                
                # Max attention (measure of focus)
                layer_head_max_attention[layer_idx, head_idx] += head_attn.max()
    
    # Average over all timesteps
    layer_head_entropy /= len(attention_history)
    layer_head_max_attention /= len(attention_history)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Entropy heatmap
    im1 = axes[0].imshow(layer_head_entropy, cmap='RdYlGn_r', aspect='auto')
    axes[0].set_xlabel('Head', fontsize=12)
    axes[0].set_ylabel('Layer', fontsize=12)
    axes[0].set_title('Average Attention Entropy\n(Higher = More Spread Out)', fontsize=14)
    axes[0].set_xticks(range(num_heads))
    axes[0].set_xticklabels(range(1, num_heads+1))
    axes[0].set_yticks(range(num_layers))
    axes[0].set_yticklabels(range(1, num_layers+1))
    plt.colorbar(im1, ax=axes[0], label='Entropy')
    
    # Max attention heatmap
    im2 = axes[1].imshow(layer_head_max_attention, cmap='RdYlGn', aspect='auto')
    axes[1].set_xlabel('Head', fontsize=12)
    axes[1].set_ylabel('Layer', fontsize=12)
    axes[1].set_title('Average Max Attention\n(Higher = More Focused)', fontsize=14)
    axes[1].set_xticks(range(num_heads))
    axes[1].set_xticklabels(range(1, num_heads+1))
    axes[1].set_yticks(range(num_layers))
    axes[1].set_yticklabels(range(1, num_layers+1))
    plt.colorbar(im2, ax=axes[1], label='Max Attention')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'layer_head_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved layer-head analysis: {save_path}")


def generate_compression_recommendations(attention_history, output_dir):
    """
    Generate specific recommendations for compression based on attention patterns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze attention patterns
    attention_to_positions = []
    
    for attention_layers in attention_history:
        if len(attention_layers) == 0:
            continue
        
        # Average over layers and heads
        all_layers = []
        for layer_attn in attention_layers:
            avg_heads = layer_attn[0].mean(axis=0)
            all_layers.append(avg_heads)
        
        avg_attn = np.mean(all_layers, axis=0)
        
        if avg_attn.shape[0] > 0:
            attention_to_positions.append(avg_attn[-1, :])
    
    if len(attention_to_positions) == 0:
        return
    
    # Compute statistics
    avg_seq_len = np.mean([len(attn) for attn in attention_to_positions])
    max_seq_len = max([len(attn) for attn in attention_to_positions])
    
    # Analyze recent vs distant attention
    recent_window = 5
    recent_attn = []
    distant_attn = []
    
    for attn in attention_to_positions:
        if len(attn) > recent_window:
            recent_attn.append(attn[-recent_window:].sum() / attn.sum())
            distant_attn.append(attn[:-recent_window].sum() / attn.sum())
    
    avg_recent = np.mean(recent_attn) if recent_attn else 0
    avg_distant = np.mean(distant_attn) if distant_attn else 0
    
    # Generate recommendations
    recommendations = []
    recommendations.append("=" * 80)
    recommendations.append("COMPRESSION STRATEGY RECOMMENDATIONS")
    recommendations.append("=" * 80)
    recommendations.append("")
    recommendations.append(f"Context Statistics:")
    recommendations.append(f"  Average sequence length: {avg_seq_len:.1f}")
    recommendations.append(f"  Maximum sequence length: {max_seq_len}")
    recommendations.append("")
    recommendations.append(f"Attention Distribution:")
    recommendations.append(f"  Recent context ({recent_window} steps): {avg_recent*100:.1f}%")
    recommendations.append(f"  Distant context: {avg_distant*100:.1f}%")
    recommendations.append("")
    
    # Strategy recommendations
    recommendations.append("Recommended Compression Strategies:")
    recommendations.append("")
    
    if avg_recent > 0.7:
        recommendations.append("1. RECENCY-BASED COMPRESSION (Recommended)")
        recommendations.append("   - The model focuses heavily on recent context")
        recommendations.append(f"   - Keep last {recent_window*2} positions, compress older ones")
        recommendations.append("   - Compress older positions with attention-weighted pooling")
        recommendations.append("")
    
    recommendations.append("2. ATTENTION-WEIGHTED COMPRESSION")
    recommendations.append("   - Identify positions with attention > threshold")
    recommendations.append("   - Keep high-attention positions, compress low-attention ones")
    recommendations.append("   - Suggested threshold: top 80-90% cumulative attention")
    recommendations.append("")
    
    recommendations.append("3. FIXED-WINDOW WITH SUMMARY")
    recommendations.append(f"   - Keep recent {recent_window*2} positions unchanged")
    recommendations.append("   - Compress older context into fixed summary tokens")
    recommendations.append("   - Use learned compression (e.g., small attention-based pooling)")
    recommendations.append("")
    
    if max_seq_len > 50:
        recommendations.append("4. HIERARCHICAL COMPRESSION")
        recommendations.append("   - Compress at multiple intervals (e.g., every 10 steps)")
        recommendations.append("   - Keep recent detail, summarize distant context")
        recommendations.append("   - Progressive compression: more aggressive for older context")
        recommendations.append("")
    
    recommendations.append("Implementation Suggestions:")
    recommendations.append("")
    recommendations.append("  a) Start simple: Fixed window + attention pooling")
    recommendations.append("  b) Compression interval: Every 10-20 steps")
    recommendations.append(f"  c) Target compressed length: {int(avg_seq_len * 0.3)}-{int(avg_seq_len * 0.5)}")
    recommendations.append("  d) Preserve query token (current state) always")
    recommendations.append("")
    recommendations.append("=" * 80)
    
    # Save recommendations
    rec_path = os.path.join(output_dir, 'compression_recommendations.txt')
    with open(rec_path, 'w') as f:
        f.write('\n'.join(recommendations))
    
    print("\n" + '\n'.join(recommendations))
    print(f"\nSaved recommendations to: {rec_path}")


if __name__ == '__main__':
    data_path = './figs/attention_analysis/attention_data.npz'
    attention_path = './figs/attention_analysis/attention_history.pkl'
    output_dir = './figs/attention_analysis'
    
    if not os.path.exists(data_path):
        print(f"Error: Attention data not found at {data_path}")
        print("\nPlease run: python evaluate_with_attention.py")
        print("This will capture attention patterns during evaluation.")
        exit(1)
    
    if not os.path.exists(attention_path):
        print(f"Error: Attention history not found at {attention_path}")
        print("\nPlease run: python evaluate_with_attention.py")
        print("This will capture attention patterns during evaluation.")
        exit(1)
    
    print("Loading attention data...")
    attention_history, rewards = load_attention_data(data_path, attention_path)
    
    print(f"Loaded {len(attention_history)} timesteps")
    print(f"Rewards - Mean: {rewards.mean():.2f}, Std: {rewards.std():.2f}")
    
    print("\nGenerating visualizations...")
    
    # Generate different visualizations
    print("\n1. Creating attention heatmaps...")
    visualize_attention_heatmaps(attention_history, output_dir, num_samples=5)
    
    print("\n2. Analyzing attention to positions...")
    visualize_attention_to_positions(attention_history, output_dir)
    
    print("\n3. Analyzing layer and head patterns...")
    analyze_layer_head_patterns(attention_history, output_dir)
    
    print("\n4. Generating compression recommendations...")
    generate_compression_recommendations(attention_history, output_dir)
    
    print("\n" + "="*80)
    print("Analysis complete! Check the following directory for results:")
    print(f"  {output_dir}")
    print("="*80)
