# Attention Visualization for History Compression

This guide helps you visualize attention patterns during Algorithm Distillation evaluation to guide the implementation of history compression.

## Overview

The attention visualization tools help you understand:
- Which parts of the history context the transformer attends to
- How attention patterns evolve during evaluation
- Compression potential (which positions can be compressed)
- Optimal compression strategies based on attention distribution

## Quick Start

### Step 1: Capture Attention During Evaluation

Run the modified evaluation script that captures attention weights:

```bash
python evaluate_with_attention.py
```

This will:
- Load your trained AD model
- Evaluate on the darkroom environment
- Capture attention weights at each timestep
- Save attention data to `./figs/attention_analysis/attention_data.npz`

**Note**: This may take longer than normal evaluation due to attention capture.

### Step 2: Analyze and Visualize Attention

Run the analysis script to generate visualizations:

```bash
python analyze_attention.py
```

This generates several visualizations in `./figs/attention_analysis/`:

1. **Attention Heatmaps** (`attention_heatmap_step*.png`)
   - Shows attention patterns at different timesteps
   - One plot per layer and head
   - Helps identify which heads focus on recent vs distant context

2. **Comprehensive Analysis** (`attention_analysis_comprehensive.png`)
   - Attention to all positions over time
   - Recent vs distant context comparison
   - Sequence length growth
   - Attention by relative position
   - Compression potential (cumulative attention)

3. **Layer-Head Analysis** (`layer_head_analysis.png`)
   - Entropy per layer/head (spread of attention)
   - Max attention per layer/head (focus level)
   - Helps identify which layers/heads are most focused

4. **Compression Recommendations** (`compression_recommendations.txt`)
   - Detailed recommendations for compression strategies
   - Statistics on attention distribution
   - Suggested compression intervals and targets

## Understanding the Visualizations

### Attention Heatmaps

- **X-axis**: Key positions (context history)
- **Y-axis**: Query positions (what attends to what)
- **Color**: Attention weight (bright = high attention)
- **Diagonal**: Self-attention within same position
- **Last row**: Most important - shows what current query attends to

### Attention to Positions Over Time

This is the MOST IMPORTANT visualization for compression:
- Shows which historical positions receive attention as evaluation progresses
- Dark vertical bands = positions that consistently receive attention (keep these!)
- Light areas = positions with low attention (candidates for compression)

### Compression Potential (Cumulative Attention)

This plot shows:
- How many positions you need to keep to retain X% of attention
- Example: If 10 positions capture 80% of attention, you can compress the rest
- Guides your compression ratio (e.g., compress 50 → 20 positions)

## Key Insights for Compression

Based on the analysis, you'll likely find:

1. **Recency Bias**: Recent positions (last 5-10 steps) get most attention
   - Strategy: Keep recent context, compress older parts

2. **Sparse Attention**: Only a few positions get high attention
   - Strategy: Attention-weighted pooling or selection

3. **Position-Specific Patterns**: Certain positions (e.g., start state) may be important
   - Strategy: Always preserve these "anchor" positions

## Recommended Compression Strategies

### Strategy 1: Fixed Window + Compression
```
Keep: Last N positions unchanged
Compress: Older positions into M summary tokens
```

### Strategy 2: Attention-Weighted Selection
```
1. Compute attention to each position
2. Keep top K positions by cumulative attention
3. Compress/discard the rest
```

### Strategy 3: Hierarchical Compression
```
Recent (last 5):  Keep all
Medium (5-20):    Compress 2:1
Distant (>20):    Compress 4:1 or more
```

## Customization

### Modify Evaluation Length

In `evaluate_with_attention.py`, adjust:
```python
eval_timesteps=min(config['horizon'] * 10, 500)  # Change 500 to desired length
```

### Change Number of Sample Heatmaps

In `analyze_attention.py`, adjust:
```python
visualize_attention_heatmaps(attention_history, output_dir, num_samples=5)  # Change 5
```

### Analyze Specific Environments

In `evaluate_with_attention.py`, change:
```python
envs = SubprocVecEnv([make_env(config, goal=test_env_args[0])])  # Change index
```

## Next Steps: Implementing Compression

After analyzing attention patterns:

1. **Choose a compression strategy** based on recommendations
2. **Modify `model/ad.py`** to add compression logic:
   ```python
   def compress_context(self, transformer_input, attention_weights):
       # Your compression logic here
       pass
   ```
3. **Add compression interval** in `evaluate_in_context()`:
   ```python
   if step % compression_interval == 0:
       transformer_input = self.compress_context(transformer_input, ...)
   ```
4. **Test inference speedup** by comparing evaluation times

## Tips

- Run analysis on multiple checkpoints to ensure patterns are consistent
- Test on different environments to verify generalization
- Start with simple compression (fixed window) before complex strategies
- Measure both speedup AND performance degradation

## Troubleshooting

**Issue**: Out of memory during attention capture
- Solution: Reduce `eval_timesteps` in `evaluate_with_attention.py`

**Issue**: Attention data file not found
- Solution: Make sure `evaluate_with_attention.py` completed successfully

**Issue**: Heatmaps look uniform
- Solution: Check if model is properly loaded and evaluating correctly

## Files Created

- `evaluate_with_attention.py`: Captures attention during evaluation
- `analyze_attention.py`: Generates all visualizations and recommendations
- `visualize_attention.py`: Additional utility functions (optional)

## Questions?

The visualizations should give you clear insights into:
- Where the model looks in the history
- How much context you can safely compress
- What compression strategy to use

Good luck implementing history compression! 🚀
