# Attention Visualization & History Compression Guide

## Overview

This package provides tools to visualize attention patterns in your Algorithm Distillation model during evaluation, helping you design an effective history compression strategy to improve inference speed.

## 🚀 Quick Start

### 1. Run the Complete Analysis Pipeline

```bash
python run_attention_analysis.py
```

This single command will:
- Capture attention patterns during evaluation
- Generate comprehensive visualizations
- Provide compression strategy recommendations

### 2. Review the Results

Check `./figs/attention_analysis/` for:
- `attention_analysis_comprehensive.png` - Main analysis (START HERE)
- `compression_recommendations.txt` - Strategy suggestions
- `attention_heatmap_step*.png` - Detailed attention at different timesteps
- `layer_head_analysis.png` - Per-layer/head patterns

### 3. Implement Compression

Use `compression_template.py` as a guide to add compression to your model.

## 📁 Files Created

| File | Purpose |
|------|---------|
| `run_attention_analysis.py` | **One-click pipeline** - Run this first! |
| `evaluate_with_attention.py` | Captures attention during evaluation |
| `analyze_attention.py` | Generates visualizations and recommendations |
| `compression_template.py` | Example compression implementation |
| `visualize_attention.py` | Additional visualization utilities |
| `ATTENTION_ANALYSIS_README.md` | Detailed documentation |

## 🎯 What You'll Learn

### 1. Where Does the Model Look?

The visualizations show:
- **Recent bias**: How much attention goes to recent vs distant history?
- **Sparse attention**: Are only a few positions important?
- **Position-specific patterns**: Do certain positions (e.g., initial state) always get attention?

### 2. Compression Potential

Key metrics:
- **Compression ratio**: How many positions needed for 80%/90% of attention?
- **Optimal window size**: How much recent context to keep?
- **Speedup potential**: Current vs compressed sequence length

### 3. Best Strategy

Based on attention patterns, you'll get recommendations like:
- "Keep last 10 positions, compress older → 3:1 ratio"
- "Use attention-weighted pooling for positions with <5% attention"
- "Implement compression every 20 steps"

## 📊 Key Visualizations Explained

### Comprehensive Analysis Plot

This 6-panel visualization is your main resource:

1. **Attention Heatmap Over Time** (top)
   - X-axis: Evaluation timestep
   - Y-axis: Context position
   - Shows which positions get attention throughout evaluation

2. **Recent vs Distant Attention** (middle-left)
   - Trend showing if model focuses on recent or distant context
   - Guides whether to use recency-based compression

3. **Sequence Length Growth** (middle-right)
   - Shows how context grows during evaluation
   - Indicates where compression is most needed

4. **Attention by Relative Position** (bottom-left)
   - Position 0 = current query, -N = N steps ago
   - Shows the "attention profile" of your model

5. **Compression Potential** (bottom-right)
   - **CRITICAL PLOT**: Shows cumulative attention curve
   - Find intersection with 80% or 90% line
   - That's how many positions you need to keep!

### Layer-Head Analysis

- **Entropy**: Higher = attention is spread out
- **Max Attention**: Higher = attention is focused
- Different heads may have different roles (some focus on recent, some on distant)

## 🛠️ Implementation Workflow

```
1. Analyze Attention
   ↓
   Run: python run_attention_analysis.py
   ↓
2. Review Results
   ↓
   Check: ./figs/attention_analysis/
   Read: compression_recommendations.txt
   ↓
3. Choose Strategy
   ↓
   Based on your attention patterns:
   - Recency-based (if recent attention > 70%)
   - Top-K selection (if sparse attention)
   - Fixed window (balanced approach)
   ↓
4. Implement Compression
   ↓
   Modify: model/ad.py
   Reference: compression_template.py
   ↓
5. Test & Iterate
   ↓
   Measure: Speedup vs Performance
   Adjust: Compression ratio/interval
```

## 💡 Common Patterns & Solutions

### Pattern 1: Strong Recency Bias
**Symptoms**: >70% attention on last 5-10 positions  
**Solution**: Fixed window compression
```python
# Keep last 10, compress rest 5:1
keep_recent = 10
compress_ratio = 0.2
```

### Pattern 2: Sparse Attention
**Symptoms**: Few positions capture >80% attention  
**Solution**: Top-K selection
```python
# Keep top 15 positions by attention
strategy = 'topk'
compression_ratio = 0.3  # Adjust based on your data
```

### Pattern 3: Uniform Attention
**Symptoms**: Attention spread evenly across history  
**Solution**: Attention-weighted pooling
```python
# Compress all positions proportionally
strategy = 'pooling'
window_size = 4  # Compress 4→1
```

## ⚙️ Configuration Options

### In `evaluate_with_attention.py`:

```python
# Evaluation length (balance between detail and speed)
eval_timesteps = min(config['horizon'] * 10, 500)

# Environment to analyze
envs = SubprocVecEnv([make_env(config, goal=test_env_args[0])])
```

### In `analyze_attention.py`:

```python
# Number of heatmaps to generate
visualize_attention_heatmaps(..., num_samples=5)

# Recent window size for analysis
recent_window = 5  # Adjust based on your domain
```

### In `compression_template.py`:

```python
COMPRESSION_CONFIG = {
    'use_compression': True,
    'compression_interval': 20,      # Steps between compressions
    'compression_ratio': 0.5,        # 0.5 = keep 50%
    'compression_strategy': 'fixed_window',  # or 'topk', 'pooling'
}
```

## 🎓 Understanding the Metrics

| Metric | Meaning | Good Value |
|--------|---------|------------|
| Positions for 80% | How many positions capture 80% of attention | <30% of total |
| Recent attention % | Attention to last 5 positions | 60-80% (typical) |
| Avg sequence length | Average context size during eval | Depends on task |
| Entropy | Spread of attention | Low = focused |

## 🔧 Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce evaluation timesteps
```python
eval_timesteps = 200  # Instead of 500
```

### Issue: Uniform Attention (No Clear Pattern)
**Possible causes**:
1. Model not fully trained
2. Task doesn't require long history
3. Model architecture issues

**Solution**: 
- Check model performance first
- Try analyzing at different training checkpoints
- Consider if compression is necessary

### Issue: Visualizations Don't Make Sense
**Check**:
1. Is the model loaded correctly?
2. Is evaluation working properly?
3. Try with different random seeds

## 📈 Measuring Success

After implementing compression, track:

```python
# Inference speed
time_without = ...
time_with = ...
speedup = time_without / time_with
print(f"Speedup: {speedup:.2f}x")

# Performance
reward_without = ...
reward_with = ...
degradation = (reward_without - reward_with) / reward_without
print(f"Performance loss: {degradation*100:.1f}%")

# Target: >2x speedup with <5% performance loss
```

## 🎯 Expected Results

Typical outcomes from compression:

| Compression Ratio | Speedup | Performance Loss |
|------------------|---------|------------------|
| 0.7 (keep 70%) | 1.2-1.5x | <1% |
| 0.5 (keep 50%) | 1.5-2.0x | 1-3% |
| 0.3 (keep 30%) | 2.0-3.0x | 3-8% |

Your results may vary based on:
- How important distant history is
- How well attention captures importance
- Compression strategy quality

## 🚦 Next Steps

1. ✅ Run attention analysis
2. ✅ Review visualizations and recommendations
3. ⬜ Choose compression strategy
4. ⬜ Implement compression in model/ad.py
5. ⬜ Test on validation set
6. ⬜ Measure speedup vs performance tradeoff
7. ⬜ Tune compression parameters
8. ⬜ Deploy optimized model

## 📚 Additional Resources

- **Detailed Guide**: See `ATTENTION_ANALYSIS_README.md`
- **Implementation Template**: See `compression_template.py`
- **Original Evaluation**: See `evaluate.py` for baseline comparison

## 🙋 Common Questions

**Q: How often should I compress?**  
A: Every 10-20 steps works well. More frequent = better compression, but more overhead.

**Q: How much can I compress?**  
A: Check the cumulative attention plot. If 20 positions capture 90% attention, you can safely compress more aggressively.

**Q: Which strategy is best?**  
A: Usually "fixed_window" - it's simple, effective, and preserves recent context.

**Q: Will compression hurt performance?**  
A: Slight degradation (1-5%) is normal. If >10%, compression is too aggressive.

**Q: Can I compress during training?**  
A: Yes! But start with inference first to validate the approach.

## 📝 Summary

This toolkit helps you:
1. **Understand** where your model looks in history
2. **Identify** compression opportunities
3. **Implement** effective compression strategies
4. **Achieve** faster inference with minimal performance loss

Start with `python run_attention_analysis.py` and follow the recommendations!

---

Good luck with your implementation! 🚀
