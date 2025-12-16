# Ablation Study Guide for AMP Prediction Ensemble

This document provides a comprehensive guide for conducting ablation studies on the Antimicrobial Peptide (AMP) Prediction Ensemble framework.

## Table of Contents
1. [Overview](#overview)
2. [Key Components for Ablation](#key-components-for-ablation)
3. [Running Ablation Studies](#running-ablation-studies)
4. [Interpreting Results](#interpreting-results)
5. [Best Practices](#best-practices)

## Overview

Ablation studies systematically evaluate the contribution of different components to the overall model performance by removing or modifying them one at a time. This helps identify:
- Which components are most critical for performance
- Redundant components that can be removed
- Optimal hyperparameter settings
- Areas for further improvement

## Key Components for Ablation

### 1. Protein Embedding Components

#### 1.1 ESM Model Variants
Test different ESM model sizes to understand the trade-off between model complexity and performance:

- **ESM-2 650M (Default)**: `facebook/esm2_t33_650M_UR50D`
  - Embedding dimension: 1280
  - Best balance of performance and efficiency
  
- **ESM-2 150M (Lightweight)**: `facebook/esm2_t30_150M_UR50D`
  - Embedding dimension: 640
  - Faster inference, lower memory footprint
  
- **ESM-2 3B (Large)**: `facebook/esm2_t36_3B_UR50D`
  - Embedding dimension: 2560
  - Highest capacity, requires more resources

**Expected Impact**: Larger models generally provide richer representations but with diminishing returns. The 650M model offers optimal cost-benefit.

#### 1.2 Embedding Types
Compare amino acid-level vs. sequence-level embeddings:

- **Amino Acid-Level Embeddings**
  - Per-residue representations (shape: [seq_len, 1280])
  - Captures positional information
  - Used by CNN, RNN, and Transformer models
  
- **Sequence-Level Embeddings**
  - Single vector per sequence (shape: [1280])
  - Obtained through pooling
  - Used by simpler classifiers (e.g., Logistic Regression)

**Expected Impact**: Amino acid-level embeddings preserve spatial information crucial for sequence-based models.

#### 1.3 Pooling Strategies (for sequence-level embeddings)
Test different pooling methods:

- **Mean Pooling**: Average over all residue embeddings
- **Max Pooling**: Take maximum value across residues
- **CLS Token**: Use the [CLS] token embedding

**Expected Impact**: Mean pooling typically works best as it preserves overall sequence information.

### 2. Model Architecture Components

#### 2.1 Individual Models
Each model contributes unique inductive biases:

- **CNN (Convolutional Neural Network)**
  - Captures local patterns and motifs
  - Multi-scale feature extraction (kernels: 3, 5, 7)
  - Expected contribution: ~15-20% of ensemble performance
  
- **BiLSTM (Bidirectional LSTM)**
  - Captures long-range dependencies
  - Bidirectional context modeling
  - Expected contribution: ~12-15% of ensemble performance
  
- **GRU (Gated Recurrent Unit)**
  - Efficient sequence modeling
  - Simpler than LSTM but effective
  - Expected contribution: ~18-22% of ensemble performance
  
- **LSTM (Long Short-Term Memory)**
  - Deep recurrent architecture (2 layers)
  - Captures complex temporal patterns
  - Expected contribution: ~15-18% of ensemble performance
  
- **BiCNN (Hybrid CNN-BiLSTM)**
  - Combines local and global features
  - Best of both worlds
  - Expected contribution: ~20-25% of ensemble performance
  
- **Transformer**
  - Self-attention mechanism
  - Captures long-range interactions
  - Expected contribution: ~10-15% of ensemble performance

#### 2.2 Model Combinations
Test different ensemble subsets:

```yaml
# Full ensemble (baseline)
models: [cnn, bilstm, gru, lstm, hybrid, transformer]

# Leave-one-out ablations
without_cnn: [bilstm, gru, lstm, hybrid, transformer]
without_bilstm: [cnn, gru, lstm, hybrid, transformer]
# ... etc for each model

# Architecture family ablations
only_recurrent: [bilstm, gru, lstm]
only_cnn_recurrent: [cnn, bilstm, gru, lstm]
minimal_ensemble: [cnn, lstm]
```

**Expected Impact**: Removing the best individual performer (typically BiCNN or GRU) should cause the largest performance drop.

### 3. Ensemble Strategy Components

#### 3.1 Voting Strategies

- **Soft Voting (Default)**
  - Averages probability predictions: `p_ensemble = (1/M) * Σ p_i`
  - Smooths predictions, reduces variance
  - Best for calibrated models
  
- **Hard Voting**
  - Majority vote on binary predictions
  - Simple and interpretable
  - Less sensitive to probability calibration
  
- **Weighted Voting**
  - Assigns weights based on validation performance
  - Optimal when individual models have varying quality
  - Weights: `w_i = 1 / (MSE_i + ε)` for regression

**Expected Impact**: Soft voting typically outperforms hard voting by 1-2% in classification tasks.

#### 3.2 Classification Threshold
Test different threshold values:

```python
thresholds = [0.5, 0.6, 0.7, 0.78, 0.8, 0.9]
```

**Expected Impact**: Optimal threshold (0.78) balances precision and recall. Higher thresholds increase precision but reduce recall.

### 4. Training Components

#### 4.1 Learning Rate
```python
learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]  # 3e-4 is default
```

**Expected Impact**: Too low → slow convergence; too high → unstable training.

#### 4.2 Dropout Rate
```python
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]  # 0.3 is default
```

**Expected Impact**: Higher dropout prevents overfitting but may underfit if too high.

#### 4.3 Batch Size
```python
batch_sizes = [32, 64, 128]  # 64 is default
```

**Expected Impact**: Larger batches → faster training but may reduce generalization.

## Running Ablation Studies

### Quick Start

```bash
# Navigate to the scripts directory
cd amp_prediction/scripts/ablation

# Run all ablation studies
python run_ablation.py \
    --config ../../configs/ablation_config.yaml \
    --study all \
    --results-dir ../../results/ablation \
    --seed 42
```

### Run Specific Studies

```bash
# Embedding ablation only
python run_ablation.py --study embedding

# Model architecture ablation only
python run_ablation.py --study model

# Ensemble strategy ablation only
python run_ablation.py --study ensemble

# Training hyperparameter ablation only
python run_ablation.py --study training
```

### Using Multiple Random Seeds

For robust results, run experiments with multiple random seeds:

```bash
for seed in 42 123 456 789 1011; do
    python run_ablation.py \
        --config ../../configs/ablation_config.yaml \
        --study all \
        --results-dir ../../results/ablation \
        --seed $seed
done
```

## Interpreting Results

### Performance Metrics

**Classification Metrics**:
- **ROC-AUC**: Overall discriminative ability (0.99+ is excellent)
- **Precision**: Positive predictive value (aim for >99%)
- **Recall**: Sensitivity (balance with precision)
- **F1-Score**: Harmonic mean of precision and recall

**Regression Metrics**:
- **Pearson R**: Linear correlation (>0.85 is strong)
- **R²**: Variance explained
- **MSE**: Mean squared error (lower is better)
- **RMSE**: Root MSE (interpretable scale)

### Comparing Performance

Calculate the **performance delta** from the baseline:

```python
delta = (ablation_metric - baseline_metric) / baseline_metric * 100
```

**Interpretation**:
- `delta < -5%`: Component is critical
- `-5% ≤ delta < -2%`: Component is important
- `-2% ≤ delta < 0%`: Component has minor impact
- `delta ≥ 0%`: Component may be redundant

### Statistical Significance

Use paired t-tests or Wilcoxon signed-rank tests to assess significance:

```python
from scipy.stats import ttest_rel

# Compare two configurations
t_stat, p_value = ttest_rel(baseline_scores, ablation_scores)

if p_value < 0.05:
    print("Difference is statistically significant")
```

## Best Practices

### 1. Experimental Design

- **Use consistent train/val/test splits** across all experiments
- **Set random seeds** for reproducibility
- **Run multiple replicates** (at least 3-5) to account for variance
- **Document all hyperparameters** for each experiment

### 2. Systematic Testing

Test components in order of expected impact:
1. Individual models (identify best performers)
2. Ensemble combinations (test leave-one-out)
3. Ensemble strategies (compare voting methods)
4. Embeddings (test different representations)
5. Training hyperparameters (fine-tune)

### 3. Resource Management

- **Embeddings**: Pre-compute and reuse to save time
- **Model checkpoints**: Save trained models for different configurations
- **Parallel experiments**: Run independent ablations in parallel
- **GPU memory**: Monitor usage, reduce batch size if needed

### 4. Result Visualization

Create informative visualizations:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Bar chart comparing model contributions
models = ['CNN', 'BiLSTM', 'GRU', 'LSTM', 'BiCNN', 'Transformer']
roc_aucs = [0.9889, 0.9869, 0.9901, 0.9905, 0.9918, 0.9905]

plt.figure(figsize=(10, 6))
plt.bar(models, roc_aucs)
plt.ylabel('ROC-AUC')
plt.title('Individual Model Performance')
plt.axhline(y=0.9939, color='r', linestyle='--', label='Ensemble')
plt.legend()
plt.savefig('individual_model_performance.png')
```

### 5. Documentation

Document your findings:
- Create a summary table with all results
- Include standard deviations
- Note any unexpected findings
- Suggest follow-up experiments

## Example Analysis Workflow

```python
# 1. Load results
import json

with open('results/ablation/model_results_20250122.json', 'r') as f:
    results = json.load(f)

# 2. Extract metrics
baseline_auc = 0.9939
ablation_aucs = {
    'without_cnn': 0.9925,
    'without_bilstm': 0.9930,
    'without_gru': 0.9910,  # Largest drop
    'without_lstm': 0.9928,
    'without_hybrid': 0.9905,  # Second largest drop
    'without_transformer': 0.9935,
}

# 3. Calculate deltas
deltas = {k: ((v - baseline_auc) / baseline_auc * 100) 
          for k, v in ablation_aucs.items()}

# 4. Rank by importance
ranked = sorted(deltas.items(), key=lambda x: x[1])
print("Component importance (by performance drop):")
for component, delta in ranked:
    print(f"{component}: {delta:.2f}%")

# Output:
# Component importance (by performance drop):
# without_hybrid: -0.34%    ← Most important
# without_gru: -0.29%       ← Second most important
# without_cnn: -0.14%
# without_lstm: -0.11%
# without_bilstm: -0.09%
# without_transformer: -0.04%  ← Least important
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce batch size
   - Use gradient accumulation
   - Test models sequentially instead of loading all at once

2. **Inconsistent Results**
   - Check random seed setting
   - Ensure data splits are identical
   - Verify model initialization

3. **Long Training Times**
   - Use pre-computed embeddings
   - Reduce number of epochs for ablation
   - Test on a validation subset first

## Further Reading

- **Ensemble Methods**: Dietterich, T. G. (2000). "Ensemble Methods in Machine Learning"
- **Ablation Studies**: Meyes, R. et al. (2019). "Ablation Studies in Artificial Neural Networks"
- **Model Interpretation**: Molnar, C. (2020). "Interpretable Machine Learning"

## Contact

For questions about ablation studies, please open an issue on GitHub or contact the maintainers.
