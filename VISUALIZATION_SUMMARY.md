# Attention Visualization Summary

## What I've Created

I've built a comprehensive attention visualization tool for your Algorithm Distillation implementation. This will help you understand how your transformer attends to historical context, which is crucial for designing an effective compression strategy.

## Files Created

1. **`visualize_attention.py`** (Main Tool)
   - Extracts attention weights from transformer layers
   - Generates multiple visualization types
   - Computes statistical metrics
   - Provides compression insights

2. **`ATTENTION_VISUALIZATION_GUIDE.md`** (Documentation)
   - Step-by-step usage instructions
   - Interpretation guide for visualizations
   - Compression strategy recommendations
   - Troubleshooting tips

3. **`setup_visualization.py`** (Setup Helper)
   - Checks and installs required dependencies (seaborn)
   - Quick setup script

## How to Use

### Step 1: Install Dependencies
```powershell
python setup_visualization.py
```

### Step 2: Run Visualization
```powershell
python visualize_attention.py
```

### Step 3: Analyze Results
Check the generated figures in `./figs/attention/` and console output for insights.

## What You'll Get

### Visualizations Generated

For each transformer layer, you'll get:

1. **Attention Heatmaps** (Per Sample)
   - Shows attention weights as 2D heatmaps
   - One subplot per attention head
   - Reveals which tokens attend to which

2. **Attention Line Plots** (Per Sample)
   - Query token's attention pattern to history
   - Average attention received by each position
   - Clear view of recency bias or uniformity

3. **Summary Statistics** (Across All Samples)
   - Distribution of mean attention weights
   - Attention concentration (std deviation)
   - Attention entropy (uniformity measure)
   - Recent vs distant attention ratio

### Console Insights

The tool automatically analyzes patterns and provides actionable insights:

```
💡 INSIGHT: Strong recency bias detected!
   Consider compressing or removing distant history.
```

or

```
💡 INSIGHT: High attention entropy (uniform distribution)!
   Model attends broadly to context. Be careful with compression.
```

## Key Features

### 1. Multi-Layer Analysis
- Analyzes all transformer layers (you have 4 layers)
- Compares attention patterns across depth
- Identifies where compression is most beneficial

### 2. Multi-Head Visualization
- Shows each attention head separately
- Reveals specialized vs generalized attention
- Helps understand head-specific patterns

### 3. Statistical Analysis
Computes critical metrics:
- **Mean Context Attention**: Overall attention to history
- **Attention Std Dev**: How focused attention is
- **Attention Entropy**: Distribution uniformity
- **Recent/Distant Ratio**: Recency bias strength
- **Max Attention Position**: Most important history point

### 4. Compression Recommendations
Based on detected patterns:
- **Recency Bias** → Keep recent steps, compress old
- **Uniform Attention** → Diverse sampling needed
- **Sparse Attention** → Keep high-attention positions only
- **Clustered Patterns** → Group similar contexts

## Understanding Your Model's Attention

### Current Architecture
- Model: AD (Algorithm Distillation)
- Transformer layers: 4
- Attention heads per layer: 4 (configurable)
- Context length: `n_transit` (from your config)
- Input: [state, action, reward, next_state] × history + query_state

### How Attention Extraction Works

The tool:
1. Loads your trained checkpoint
2. Processes test samples through the model
3. Intercepts attention weights at each layer
4. Analyzes attention from query token to history
5. Computes statistics across multiple samples
6. Generates visualizations and insights

### Causal Masking

Your model uses causal masking (autoregressive), which means:
- Tokens can only attend to previous positions
- This prevents information leakage
- Attention matrices will have a triangular pattern
- The visualization tool respects this constraint

## Compression Strategy Design Workflow

1. **Run Visualization**
   ```powershell
   python visualize_attention.py
   ```

2. **Identify Pattern**
   - Look at console output
   - Check Recent/Distant ratio
   - Review attention entropy
   - Examine heatmaps

3. **Choose Strategy**

   **If Recent/Distant > 2.0:**
   ```python
   # Keep recent K steps only
   compressed_history = history[-K:]
   ```

   **If Entropy > 2.0:**
   ```python
   # Uniform sampling to preserve diversity
   indices = np.linspace(0, len(history)-1, K).astype(int)
   compressed_history = history[indices]
   ```

   **If Sparse (few high-attention positions):**
   ```python
   # Keep top-K attended positions
   top_k_indices = attention_scores.topk(K).indices
   compressed_history = history[top_k_indices]
   ```

4. **Implement Compression**
   - Modify `ad.py` forward method
   - Add compression logic before transformer
   - Test on validation set

5. **Validate**
   - Compare performance: compressed vs full
   - Check if task accuracy is maintained
   - Measure speedup and memory savings

## Expected Insights

Based on Algorithm Distillation papers, you might see:

### Early Layers
- Broader, more uniform attention
- Less recency bias
- Useful for understanding overall context importance

### Later Layers
- More focused, targeted attention
- Stronger recency bias
- Critical for final prediction

### Common Patterns
- **Recency bias**: Last few steps get most attention
- **Landmark positions**: Some positions consistently attended
- **Head specialization**: Different heads focus on different aspects

## Next Steps After Visualization

### 1. Immediate
- Run the tool and review visualizations
- Identify dominant attention pattern
- Note the Recent/Distant ratio

### 2. Short-term
- Design compression strategy based on patterns
- Implement compression in model forward pass
- Test on a small subset of data

### 3. Long-term
- Train compressed model from scratch
- Compare performance metrics
- Iterate on compression strategy
- Analyze compression vs performance tradeoff

## Technical Details

### Attention Extraction Method
The tool uses forward hooks to capture attention weights during inference:
```python
# Attention weights shape: (batch, num_heads, seq_len, seq_len)
# - batch: number of samples
# - num_heads: 4 in your model
# - seq_len: n_transit (context) + 1 (query)
# - Last dimension: attention distribution over keys
```

### Query Token Analysis
The query state token (last position) is most important:
- It's used for final action prediction
- Its attention pattern reveals what history matters
- Focus on the last row of attention matrices

### Statistics Computation
- **Mean**: Average attention weight to context
- **Std**: Concentration measure (high = focused)
- **Entropy**: -Σ(p * log(p)), uniformity measure
- **Ratio**: mean(recent 25%) / mean(distant 25%)

## Customization Options

### Change Number of Samples
Edit line 431 in `visualize_attention.py`:
```python
num_samples = min(10, len(test_dataset))  # Default is 5
```

### Analyze Specific Layer
Run analysis for just one layer:
```python
# In main(), replace the for loop:
layer_idx = 0  # Analyze only layer 0
extract_and_visualize_attention(config, model, test_dataloader,
                               num_samples=5, layer_idx=layer_idx)
```

### Change Figure Size
Modify `figsize` parameters in visualization functions:
```python
visualize_attention_heatmap(..., figsize=(16, 12))  # Larger figures
```

## Performance Considerations

- **Memory**: Attention extraction requires storing weights
- **Time**: ~1-2 minutes per layer for 5 samples
- **Storage**: Each figure is ~200-500 KB
- **Recommendation**: Start with 5 samples, increase if needed

## Troubleshooting

### "No checkpoint found"
- Ensure you've trained the model: `python train.py`
- Check the `runs/` directory for checkpoints

### Out of memory
- Reduce `num_samples` to 2-3
- Process one layer at a time
- Use smaller batch size (already set to 1)

### Figures not displaying
- Figures are saved even if display fails
- Check `./figs/attention/` directory
- Open PNG files manually

## Example Output Interpretation

```
Sample 1:
  Sequence length: 20
  Context tokens: 19
  Mean context attention: 0.0526  # ~5% per token on average
  Attention std dev: 0.0234       # Moderate concentration
  Max attention position: 18      # Attends most to last context token
  Recent/Distant ratio: 4.23      # Strong recency bias!
  Attention entropy: 1.8934       # Moderate uniformity

💡 INSIGHT: Strong recency bias detected!
   Consider compressing or removing distant history.
```

**Interpretation**: 
- Model heavily favors recent history (ratio = 4.23)
- Most attention to position 18 (second-to-last)
- Good candidate for recency-based compression
- Can safely remove or compress distant history (positions 0-10)

**Compression Recommendation**:
Keep last 10 steps, summarize or discard earlier history.

## Research Questions This Answers

1. ✅ **Does the model exhibit recency bias?**
   - Check Recent/Distant ratio
   
2. ✅ **Is attention uniform or focused?**
   - Check attention entropy
   
3. ✅ **Which history positions matter most?**
   - Check max attention position and heatmaps
   
4. ✅ **Are attention heads specialized?**
   - Compare per-head heatmaps
   
5. ✅ **How does attention change across layers?**
   - Compare statistics across layers

## Resources

- Original AD Paper: https://arxiv.org/abs/2210.14215
- Transformer Attention: Vaswani et al., "Attention Is All You Need"
- Compression Techniques: Various context compression papers

## Support

If you encounter issues:
1. Check the guide: `ATTENTION_VISUALIZATION_GUIDE.md`
2. Review console output for error messages
3. Verify checkpoint exists in `runs/` directory
4. Ensure all dependencies are installed

Good luck with your compression design! The attention patterns will reveal exactly which parts of the history your model relies on. 🚀
