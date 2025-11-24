# Transformer Attention Visualization for Algorithm Distillation

This toolkit helps you visualize and analyze transformer attention patterns in your Algorithm Distillation implementation, enabling you to design effective history compression strategies.

## 🎯 Purpose

Algorithm Distillation uses transformer models to process historical trajectories. However, many historical contexts are similar and redundant. By visualizing attention patterns, you can:

1. **Understand** what history the model actually uses
2. **Identify** redundant or low-importance positions
3. **Design** targeted compression strategies
4. **Reduce** computational cost and memory usage
5. **Maintain** model performance with less context

## 📦 What's Included

### Core Tools

- **`visualize_attention.py`** - Main visualization tool
  - Extracts attention weights from trained models
  - Generates heatmaps, line plots, and statistics
  - Provides automatic insights for compression design
  
- **`compression_strategies.py`** - Ready-to-use compression methods
  - RecencyCompressor (for recency bias)
  - UniformCompressor (for uniform attention)
  - AttentionBasedCompressor (for sparse attention)
  - AdaptiveCompressor (automatically selects strategy)
  - ThresholdCompressor (keeps high-attention positions)
  - ExponentialDecayCompressor (balanced approach)

### Helper Scripts

- **`quickstart.py`** - Interactive setup and verification
- **`setup_visualization.py`** - Dependency installer

### Documentation

- **`ATTENTION_VISUALIZATION_GUIDE.md`** - Comprehensive usage guide
- **`VISUALIZATION_SUMMARY.md`** - Technical details and examples
- **`COMPRESSION_GUIDE.md`** - This file

## 🚀 Quick Start

### 1. Verify Setup
```bash
python quickstart.py
```

This checks:
- ✓ All dependencies installed
- ✓ Trained model available
- ✓ Dataset ready

### 2. Install Missing Dependencies (if needed)
```bash
python setup_visualization.py
```

or manually:
```bash
pip install seaborn
```

### 3. Run Attention Visualization
```bash
python visualize_attention.py
```

**Expected time**: 5-10 minutes for 4 layers × 5 samples

**Output location**: `./figs/attention/layer_*/`

### 4. Test Compression Methods
```bash
python compression_strategies.py
```

This demonstrates all available compression strategies with dummy data.

## 📊 Understanding the Visualizations

### Generated Figures

For each transformer layer, you'll get:

#### 1. Attention Heatmaps (`attention_heatmap_sample*_layer*.png`)
- **Shows**: Attention weights between all token pairs
- **Format**: One subplot per attention head
- **Interpretation**:
  - Bright areas = strong attention
  - Last row = query token's attention to history
  - Triangular pattern = causal masking

#### 2. Attention Line Plots (`attention_lines_sample*_layer*.png`)
- **Left plot**: Query token attention distribution
- **Right plot**: Average attention to each position
- **Interpretation**:
  - Peaks indicate important positions
  - Decay patterns show recency bias
  - Flat distribution = uniform attention

#### 3. Summary Statistics (`attention_summary_layer*.png`)
- **4 histograms** showing distribution across samples:
  - Mean context attention
  - Attention concentration (std dev)
  - Attention entropy (uniformity)
  - Recent/Distant ratio

### Key Metrics Explained

**Mean Context Attention**
- Average attention weight to historical context
- Higher = more context utilization
- Lower = more focused on query

**Attention Std Dev**
- Measures concentration vs spread
- High = focused on few positions
- Low = spread across many positions

**Attention Entropy**
- Uniformity measure
- High (>2.0) = uniform distribution
- Low (<1.5) = peaked distribution

**Recent/Distant Ratio**
- Compares attention to recent vs distant history
- High (>2.0) = strong recency bias
- Low (<1.2) = uniform temporal attention

## 🎨 Compression Strategy Selection

Based on visualization results:

### Pattern 1: Strong Recency Bias
**Indicators:**
- Recent/Distant ratio > 2.0
- Attention concentrated on last few positions
- Exponential decay in line plots

**Recommended Strategy:**
```python
from compression_strategies import RecencyCompressor
compressor = RecencyCompressor(keep_recent=10)
```

**Compression approach**: Keep only recent K timesteps, discard old history

---

### Pattern 2: Uniform Attention
**Indicators:**
- Attention entropy > 2.0
- Flat attention distribution
- No clear peaks in line plots

**Recommended Strategy:**
```python
from compression_strategies import UniformCompressor
compressor = UniformCompressor(target_length=10)
```

**Compression approach**: Sample evenly across history to preserve diversity

---

### Pattern 3: Sparse Focused Attention
**Indicators:**
- Low entropy < 1.5
- Few clear peaks in heatmap
- High std deviation

**Recommended Strategy:**
```python
from compression_strategies import AttentionBasedCompressor
compressor = AttentionBasedCompressor(target_length=10)
```

**Compression approach**: Keep only high-attention positions

---

### Pattern 4: Complex/Mixed Patterns
**Indicators:**
- Different patterns across layers
- Moderate entropy (1.5-2.0)
- Varied attention distributions

**Recommended Strategy:**
```python
from compression_strategies import AdaptiveCompressor
compressor = AdaptiveCompressor(target_length=10)
```

**Compression approach**: Automatically adapt based on attention statistics

## 🔧 Integration Guide

### Step 1: Choose Your Compressor

Based on visualization results, select appropriate strategy from `compression_strategies.py`.

### Step 2: Modify Your Model

Edit `model/ad.py`:

```python
# Add at the top
from compression_strategies import RecencyCompressor  # or your chosen strategy

# In AD.__init__
def __init__(self, config):
    super(AD, self).__init__()
    # ... existing code ...
    
    # Add compressor
    self.compressor = RecencyCompressor(keep_recent=10)

# In AD.forward, before transformer
def forward(self, x):
    # ... existing code to get states, actions, rewards, next_states ...
    
    # Apply compression BEFORE embedding
    states, actions, rewards, next_states = self.compressor.compress(
        states, actions, rewards, next_states
    )
    
    # Continue with existing code
    context, _ = pack([states, actions, rewards, next_states], 'b n *')
    # ... rest of forward pass ...
```

### Step 3: Adjust Configuration

Update your config file (e.g., `config/model/ad_dr.yaml`):

```yaml
# Original
n_transit: 50  # or whatever you had

# Compressed
n_transit: 10  # matching your compression target
```

### Step 4: Retrain and Evaluate

```bash
# Train with compression
python train.py

# Evaluate performance
python evaluate.py
```

## 📈 Expected Benefits

### Memory Reduction
- **50% compression**: ~50% memory savings
- **80% compression**: ~80% memory savings
- Direct correlation to target_length/original_length

### Speed Improvement
- Transformer complexity: O(n²) in sequence length
- **50% compression**: ~75% compute reduction
- **80% compression**: ~96% compute reduction

### Performance Impact
- Well-designed compression: <5% accuracy loss
- Poorly designed: >10% accuracy loss
- Key: Match compression to attention patterns!

## 🔍 Troubleshooting

### "No checkpoint found"
**Solution**: Train your model first with `python train.py`

### Out of memory during visualization
**Solution**: Edit `visualize_attention.py` line 431:
```python
num_samples = min(2, len(test_dataset))  # Reduce from 5 to 2
```

### Figures not showing
**Solution**: Figures are saved to disk. Check `./figs/attention/` directory

### Poor compression results
**Solution**: 
1. Re-run visualization with more samples
2. Verify pattern interpretation
3. Try different compression strategies
4. Adjust compression ratio (keep more history)

## 📚 Additional Resources

### Documentation Files
- `ATTENTION_VISUALIZATION_GUIDE.md` - Detailed usage instructions
- `VISUALIZATION_SUMMARY.md` - Technical implementation details

### Code Examples
- `visualize_attention.py` - Full visualization implementation
- `compression_strategies.py` - All compression methods with examples

### Original Paper
- Algorithm Distillation: https://arxiv.org/abs/2210.14215

## 🎓 Understanding Your Results

### Sample Console Output

```
ANALYZING LAYER 0
================

Sample 1:
  Sequence length: 20
  Context tokens: 19
  Mean context attention: 0.0526
  Attention std dev: 0.0234
  Max attention position: 18
  Recent/Distant ratio: 4.23
  Attention entropy: 1.89

💡 INSIGHT: Strong recency bias detected!
   Consider compressing or removing distant history.
```

**Interpretation:**
- Model focuses on recent history (ratio 4.23)
- Position 18 (second-to-last) gets most attention
- Moderate entropy suggests some spread
- **Recommendation**: Use RecencyCompressor, keep last 10-12 steps

## 🤝 Contributing

Feel free to extend this toolkit:
- Add new compression strategies
- Improve visualizations
- Optimize performance
- Add new metrics

## 📝 Citation

If this visualization toolkit helps your research:

```bibtex
@misc{ad_compression_toolkit,
  title={Attention Visualization Toolkit for Algorithm Distillation},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/your-repo}}
}
```

## ❓ FAQ

**Q: How long does visualization take?**
A: ~5-10 minutes for 4 layers with 5 samples each. Scales with num_layers × num_samples.

**Q: Can I analyze specific sequences?**
A: Yes, modify the dataset filtering in `visualize_attention.py` main function.

**Q: Which layer should I focus on?**
A: Usually the last layer, as it directly influences predictions. But compare all layers for insights.

**Q: How much can I compress safely?**
A: Start with 50% compression (keep half the history). Evaluate and adjust based on performance.

**Q: What if compression hurts performance?**
A: 1) Keep more history, 2) Try different strategy, 3) Use adaptive compression, 4) Consider attention patterns from different layers.

## 🎯 Success Metrics

After implementing compression, track:
- ✓ Training/inference speed improvement
- ✓ Memory usage reduction  
- ✓ Model accuracy (should stay within 95%+ of baseline)
- ✓ Compression ratio achieved
- ✓ Attention pattern changes (re-run visualization)

## 🌟 Best Practices

1. **Visualize First**: Always analyze attention before compressing
2. **Start Conservative**: Begin with 50% compression, increase gradually
3. **Validate Thoroughly**: Test on multiple scenarios/goals
4. **Monitor Performance**: Track accuracy, reward, and convergence
5. **Iterate**: Re-visualize after compression to verify patterns

---

**Ready to start?** Run `python quickstart.py` to begin! 🚀
