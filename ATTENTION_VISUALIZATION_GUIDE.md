# Attention Visualization Guide

This guide explains how to visualize transformer attention patterns in your Algorithm Distillation implementation to design an effective compression strategy.

## Quick Start

### 1. Install Additional Dependencies (if needed)

```bash
pip install seaborn
```

### 2. Run the Visualization Tool

```bash
python visualize_attention.py
```

This will:
- Load your trained model from the latest checkpoint
- Extract attention weights from all transformer layers
- Generate comprehensive visualizations
- Provide insights for compression strategy design

## Output Files

The tool creates visualizations in `./figs/attention/` organized by layer:

```
./figs/attention/
├── layer_0/
│   ├── attention_heatmap_sample0_layer0.png
│   ├── attention_lines_sample0_layer0.png
│   ├── attention_heatmap_sample1_layer0.png
│   ├── ...
│   └── attention_summary_layer0.png
├── layer_1/
│   └── ...
├── layer_2/
│   └── ...
└── layer_3/
    └── ...
```

## Understanding the Visualizations

### 1. Attention Heatmaps
- **What**: Shows attention weights between all token pairs
- **Interpretation**:
  - Bright areas = strong attention
  - Each row shows what a query token attends to
  - Last row = query state's attention to history
  - Multiple subplots = different attention heads

### 2. Attention Line Plots
- **Left plot**: Query token's attention to all history positions
- **Right plot**: Average attention each position receives
- **Key insight**: Identifies which historical positions matter most

### 3. Summary Statistics
Four key metrics across all samples:
1. **Mean Context Attention**: Average attention weight to history
2. **Attention Concentration (Std Dev)**: How focused vs spread out attention is
3. **Attention Entropy**: Uniformity of attention distribution
4. **Recent/Distant Ratio**: Recency bias in attention

## Compression Strategy Design

Based on attention patterns, consider these compression approaches:

### Pattern 1: Strong Recency Bias (Recent/Distant Ratio > 2.0)
**What it means**: Model focuses on recent history, ignores distant past

**Compression strategies**:
- ✅ Keep recent K timesteps fully
- ✅ Compress or remove distant history
- ✅ Use exponential decay for history retention
- Example: Keep last 10 steps, summarize older steps

### Pattern 2: Uniform Attention (High Entropy > 2.0)
**What it means**: Model attends broadly across all history

**Compression strategies**:
- ✅ Use diverse sampling to preserve coverage
- ✅ K-means clustering to maintain representativeness
- ⚠️ Be careful with aggressive compression
- Example: Sample evenly spaced history points

### Pattern 3: Focused Attention (Low Entropy < 1.5)
**What it means**: Model has clear attention preferences

**Compression strategies**:
- ✅ Keep high-attention positions
- ✅ Prune low-attention history aggressively
- ✅ Use attention scores for importance sampling
- Example: Keep top 30% attended positions

### Pattern 4: Sparse Attention
**What it means**: Only a few positions receive high attention

**Compression strategies**:
- ✅ Threshold-based filtering (keep attention > 0.1)
- ✅ Top-K selection per query
- ✅ Landmark-based compression
- Example: Keep 5 most attended positions + query

## Key Metrics to Check

After running the visualization, look for these insights in the console output:

```
OVERALL ATTENTION STATISTICS
====================================
Mean context attention: 0.0234 ± 0.0056
Attention concentration (std): 0.0123 ± 0.0034
Attention entropy: 1.7834 ± 0.2341
Recent/Distant ratio: 3.45 ± 0.87

💡 INSIGHT: Strong recency bias detected!
   Consider compressing or removing distant history.
```

## Example Compression Implementation

Based on attention analysis, here's a template for compression:

```python
def compress_history(states, actions, rewards, next_states, attention_weights, k=10):
    """
    Compress history based on attention patterns.
    
    Args:
        states, actions, rewards, next_states: History tensors
        attention_weights: Attention scores for each position
        k: Number of history steps to keep
    
    Returns:
        Compressed history tensors
    """
    # Strategy 1: Keep recent K steps
    if RECENCY_BIAS:
        return states[-k:], actions[-k:], rewards[-k:], next_states[-k:]
    
    # Strategy 2: Keep top-K attended positions
    elif SPARSE_ATTENTION:
        top_indices = torch.topk(attention_weights, k).indices
        return (states[top_indices], actions[top_indices], 
                rewards[top_indices], next_states[top_indices])
    
    # Strategy 3: Uniform sampling
    elif UNIFORM_ATTENTION:
        indices = torch.linspace(0, len(states)-1, k).long()
        return (states[indices], actions[indices], 
                rewards[indices], next_states[indices])
```

## Advanced Analysis

### Analyze Multiple Layers
The tool automatically analyzes all transformer layers. Compare patterns across layers:
- Early layers: Often show broader attention
- Late layers: Often show more focused attention
- Use insights from the layer that matters most for your task

### Modify Number of Samples
Edit `visualize_attention.py` line 431:
```python
num_samples = min(10, len(test_dataset))  # Increase to 10 or more
```

### Analyze Specific Sequences
To focus on particular scenarios, modify the dataset filtering in the main function.

## Troubleshooting

### Issue: "No checkpoint found"
**Solution**: Train your model first using `python train.py`

### Issue: Out of memory
**Solution**: Reduce `num_samples` in the main function (line 431)

### Issue: Slow execution
**Solution**: Attention extraction is compute-intensive. This is normal for transformer analysis.

## Next Steps

1. ✅ Run `python visualize_attention.py`
2. 📊 Analyze the generated visualizations
3. 💡 Identify attention patterns (recency bias, sparsity, etc.)
4. 🔧 Design compression strategy based on patterns
5. 🧪 Implement and test compression in your AD model
6. 📈 Compare performance: compressed vs full history

## Citation

If you use this visualization tool in your research, consider citing:

```bibtex
@article{laskin2022context,
  title={In-context Reinforcement Learning with Algorithm Distillation},
  author={Laskin, Michael and Wang, Luyu and Oh, Junhyuk and Parisotto, Emilio and Spencer, Stephen and Steigerwald, Richie and Strouse, DJ and Hansen, Steven and Filos, Angelos and Brooks, Ethan and others},
  journal={arXiv preprint arXiv:2210.14215},
  year={2022}
}
```

## Questions?

The tool provides detailed console output explaining each metric and suggesting compression strategies. Review the console output carefully after running the visualization.

Good luck with your compression design! 🚀
