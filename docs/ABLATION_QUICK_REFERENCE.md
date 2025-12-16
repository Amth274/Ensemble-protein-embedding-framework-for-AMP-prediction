# Quick Reference: Ablation Studies

## Quick Start

```bash
# Navigate to ablation scripts
cd amp_prediction/scripts/ablation

# Run all ablation studies
python run_ablation.py --config ../../configs/ablation_config.yaml --study all

# Run specific study
python run_ablation.py --study model --seed 42
```

## Component Categories

| Category | Count | Examples |
|----------|-------|----------|
| 🧬 Embedding Variants | 3 | ESM-650M, ESM-150M, ESM-3B |
| 🔀 Pooling Strategies | 3 | Mean, Max, CLS |
| 🧠 Model Architectures | 6 | CNN, BiLSTM, GRU, LSTM, BiCNN, Transformer |
| 📊 Model Combinations | 10 | Full, Leave-one-out (6), Subsets (3) |
| 🗳️ Voting Strategies | 4 | Soft, Hard, Weighted, Adaptive |
| 🎯 Classification Thresholds | 6 | 0.5, 0.6, 0.7, 0.78, 0.8, 0.9 |
| 📚 Learning Rates | 4 | 1e-4, 3e-4, 5e-4, 1e-3 |
| 💧 Dropout Rates | 5 | 0.1, 0.2, 0.3, 0.4, 0.5 |
| 📦 Batch Sizes | 3 | 32, 64, 128 |

## Total Experiments: 40+ configurations

## Key Files

```
amp_prediction/
├── configs/
│   ├── config.yaml                    # Base configuration
│   └── ablation_config.yaml           # Ablation study config ⭐
├── scripts/
│   └── ablation/
│       ├── run_ablation.py            # Main runner script ⭐
│       ├── ablation_utils.py          # Utility functions
│       └── __init__.py
└── docs/
    ├── ABLATION_GUIDE.md              # Comprehensive guide ⭐
    └── ABLATION_COMPONENTS_SUMMARY.md # Component summary ⭐
```

## Command Cheat Sheet

```bash
# All ablations
python run_ablation.py --study all

# Model architecture ablation
python run_ablation.py --study model

# Ensemble strategy ablation
python run_ablation.py --study ensemble

# Training hyperparameters
python run_ablation.py --study training

# Embedding ablation
python run_ablation.py --study embedding

# Custom results directory
python run_ablation.py --study all --results-dir /path/to/results

# Multiple seeds for robustness
for seed in 42 123 456; do
    python run_ablation.py --study all --seed $seed
done

# Generate all experiment configs
python ablation_utils.py \
    ../../configs/config.yaml \
    ../../configs/ablation_config.yaml \
    ../../configs/ablation_experiments
```

## Expected Performance Impact

| Component Removed | Expected ΔRoc-AUC | Criticality |
|-------------------|-------------------|-------------|
| BiCNN | -0.30% to -0.40% | ⚠️ Critical |
| GRU | -0.25% to -0.35% | ⚠️ Critical |
| CNN | -0.10% to -0.20% | ⚡ Important |
| LSTM | -0.10% to -0.15% | ⚡ Important |
| BiLSTM | -0.05% to -0.10% | ✓ Moderate |
| Transformer | -0.02% to -0.05% | ✓ Minor |

## Output Structure

```
results/ablation/
├── embedding_results_20250122_103045.json
├── embedding_summary_20250122_103045.txt
├── model_results_20250122_103120.json
├── model_summary_20250122_103120.txt
├── ensemble_results_20250122_103145.json
├── ensemble_summary_20250122_103145.txt
├── training_results_20250122_103210.json
├── training_summary_20250122_103210.txt
└── all_ablations_results_20250122_103230.json
```

## Metrics to Track

### Classification
- ✅ ROC-AUC (primary)
- ✅ Precision
- ✅ Recall
- ✅ F1-Score
- ✅ Accuracy

### Regression
- ✅ Pearson R (primary)
- ✅ R²
- ✅ MSE
- ✅ RMSE
- ✅ MAE

## Statistical Significance

Use paired t-tests to determine if differences are significant:

```python
from scipy.stats import ttest_rel

# Compare two configurations
t_stat, p_value = ttest_rel(baseline_scores, ablation_scores)
print(f"Significant: {p_value < 0.05}")
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| ModuleNotFoundError | Install dependencies: `pip install -e .` |
| Out of Memory | Reduce batch size or test fewer models |
| Slow execution | Pre-compute embeddings, use subset of data |
| Inconsistent results | Set random seed, check data splits |

## Best Practices

1. ✅ Test one component at a time
2. ✅ Use consistent random seeds
3. ✅ Run multiple replicates (3-5)
4. ✅ Pre-compute embeddings
5. ✅ Save all configurations
6. ✅ Document unexpected results

## Need Help?

- 📖 Read: `docs/ABLATION_GUIDE.md`
- 📋 Check: `docs/ABLATION_COMPONENTS_SUMMARY.md`
- 🔧 Example: See example workflows in the guide
- 🐛 Issues: Open GitHub issue

---

**Version**: 1.0 | **Updated**: 2025-01-22
