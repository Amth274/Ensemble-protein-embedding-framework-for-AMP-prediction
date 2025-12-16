# Key Components Identified for Ablation Studies

This document summarizes all key components of the AMP Prediction Ensemble that have been identified for systematic ablation studies.

## 1. Protein Embedding Components

### 1.1 ESM Model Variants
| Component | Model Path | Embedding Dim | Description |
|-----------|------------|---------------|-------------|
| ESM-2 650M (Default) | `facebook/esm2_t33_650M_UR50D` | 1280 | Optimal balance of performance and efficiency |
| ESM-2 150M | `facebook/esm2_t30_150M_UR50D` | 640 | Lightweight version for faster inference |
| ESM-2 3B | `facebook/esm2_t36_3B_UR50D` | 2560 | Largest model with highest capacity |

### 1.2 Embedding Types
| Component | Description | Shape | Use Case |
|-----------|-------------|-------|----------|
| Amino Acid-Level | Per-residue embeddings | `[seq_len, 1280]` | Sequence models (CNN, RNN, Transformer) |
| Sequence-Level (Mean) | Mean pooling over residues | `[1280]` | Simple classifiers |
| Sequence-Level (Max) | Max pooling over residues | `[1280]` | Simple classifiers |
| Sequence-Level (CLS) | CLS token embedding | `[1280]` | Simple classifiers |

### 1.3 Pooling Strategies
- Mean pooling (default)
- Max pooling
- CLS token

## 2. Model Architecture Components

### 2.1 Individual Models in Ensemble

| Model | Class Name | Description | Key Features |
|-------|------------|-------------|--------------|
| CNN | `CNN1DAMPClassifier` | 1D CNN with multi-scale convolutions | 3 conv layers (kernels: 3, 5, 7) |
| BiLSTM | `AMPBilstmClassifier` | Bidirectional LSTM | 256 hidden units, 1 layer |
| GRU | `GRUClassifier` | Bidirectional GRU | 256 hidden units, 1 layer |
| LSTM | `AMP_BiRNN` | 2-layer Bidirectional RNN | 256 hidden units, 2 layers |
| BiCNN | `CNN_BiLSTM_Classifier` | Hybrid CNN + BiLSTM | Combines local and global features |
| Transformer | `AMPTransformerClassifier` | Transformer encoder | Multi-head attention, 1 head, 1 layer |

### 2.2 Model Combinations for Ablation

| Configuration | Models Included | Purpose |
|---------------|-----------------|---------|
| Full Ensemble | All 6 models | Baseline |
| Without CNN | BiLSTM, GRU, LSTM, BiCNN, Transformer | Leave-one-out: CNN |
| Without BiLSTM | CNN, GRU, LSTM, BiCNN, Transformer | Leave-one-out: BiLSTM |
| Without GRU | CNN, BiLSTM, LSTM, BiCNN, Transformer | Leave-one-out: GRU |
| Without LSTM | CNN, BiLSTM, GRU, BiCNN, Transformer | Leave-one-out: LSTM |
| Without BiCNN | CNN, BiLSTM, GRU, LSTM, Transformer | Leave-one-out: BiCNN |
| Without Transformer | CNN, BiLSTM, GRU, LSTM, BiCNN | Leave-one-out: Transformer |
| Only Recurrent | BiLSTM, GRU, LSTM | Recurrent models only |
| Only CNN+Recurrent | CNN, BiLSTM, GRU, LSTM | Traditional architectures |
| Minimal Ensemble | CNN, LSTM | Minimal viable ensemble |

### 2.3 Model Hyperparameters

**Common across all models:**
- Embedding dimension: 1280
- Dropout rate: 0.3 (default)

**Model-specific:**
- CNN: Filter sizes [512, 256, 128], Kernel sizes [3, 5, 7]
- BiLSTM/GRU/LSTM: Hidden dim 256, 1-2 layers
- BiCNN: CNN channels 256, LSTM hidden 128
- Transformer: 1 head, 1 layer

## 3. Ensemble Strategy Components

### 3.1 Voting Strategies

| Strategy | Class | Description | Parameters |
|----------|-------|-------------|------------|
| Soft Voting | `SoftVoting` | Averages probability predictions | threshold=0.78 |
| Hard Voting | `HardVoting` | Majority voting on binary predictions | threshold=0.5 |
| Weighted Voting | `WeightedVoting` | Weighted average by validation performance | threshold=0.78, weight_method='inverse_mse' |
| Adaptive Voting | `AdaptiveVoting` | Confidence-weighted adaptive voting | base_strategy='soft_voting' |

### 3.2 Classification Thresholds
Test thresholds: [0.5, 0.6, 0.7, 0.78 (optimal), 0.8, 0.9]

### 3.3 Regression Weight Methods
- Equal weights
- Inverse MSE (default)
- Inverse MAE
- R²-based

## 4. Training Components

### 4.1 Learning Rates
Test values: [1e-4, 3e-4 (default), 5e-4, 1e-3]

### 4.2 Dropout Rates
Test values: [0.1, 0.2, 0.3 (default), 0.4, 0.5]

### 4.3 Batch Sizes
Test values: [32, 64 (default), 128]

### 4.4 Optimizers
- Adam (default)
- AdamW (with weight decay)
- SGD (with momentum)

### 4.5 Learning Rate Schedulers
- Cosine annealing (default)
- Step decay
- Reduce on plateau
- None

## 5. Data Components

### 5.1 Sequence Length Handling
Test values: [50, 75, 100 (default), 150]

### 5.2 Data Augmentation
- None (default)
- Random mutations
- Random truncation
- Combined strategies

### 5.3 Train/Val/Test Split Ratios
- 70/15/15
- 80/10/10 (default)
- 60/20/20

## Files Created for Ablation Studies

### Configuration Files
1. **`configs/ablation_config.yaml`**
   - Comprehensive configuration documenting all ablation components
   - Organized into 5 main sections: embeddings, models, ensembles, training, data
   - Includes experimental design guidelines

### Scripts
2. **`scripts/ablation/run_ablation.py`**
   - Main script for running ablation studies
   - Supports running all or specific ablation types
   - Generates structured results and summaries
   - Command-line interface for easy execution

3. **`scripts/ablation/ablation_utils.py`**
   - Utility functions for configuration management
   - `AblationConfigManager` class for modifying configs
   - Functions to generate all experiment configs automatically
   - Config comparison utilities

4. **`scripts/ablation/__init__.py`**
   - Module initialization

### Documentation
5. **`docs/ABLATION_GUIDE.md`**
   - Comprehensive guide for conducting ablation studies
   - Detailed explanations of each component
   - Instructions for running experiments
   - Best practices and troubleshooting
   - Example analysis workflows

6. **`docs/ABLATION_COMPONENTS_SUMMARY.md`** (this file)
   - Quick reference for all identified components
   - Summary tables for easy lookup

## Usage Examples

### Run All Ablation Studies
```bash
cd amp_prediction/scripts/ablation
python run_ablation.py \
    --config ../../configs/ablation_config.yaml \
    --study all \
    --results-dir ../../results/ablation \
    --seed 42
```

### Run Specific Ablation Study
```bash
# Model architecture ablation only
python run_ablation.py --study model --seed 42

# Ensemble strategy ablation only
python run_ablation.py --study ensemble --seed 42

# Training hyperparameter ablation only
python run_ablation.py --study training --seed 42

# Embedding ablation only
python run_ablation.py --study embedding --seed 42
```

### Generate All Experiment Configurations
```bash
python ablation_utils.py \
    ../../configs/config.yaml \
    ../../configs/ablation_config.yaml \
    ../../configs/ablation_experiments
```

## Expected Outcomes

### Performance Ranking (based on expected contributions)
1. **Most Critical**: BiCNN (hybrid architecture)
2. **Very Important**: GRU, CNN
3. **Important**: LSTM, BiLSTM
4. **Moderately Important**: Transformer
5. **Ensemble Strategy**: Soft voting > Weighted > Hard
6. **Embeddings**: ESM-650M optimal, amino-acid level > sequence-level

### Metrics to Track

**Classification:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC (primary metric)
- PR-AUC

**Regression:**
- MSE
- RMSE
- MAE
- R²
- Pearson R (primary metric)
- Spearman R

## Integration with Existing Codebase

The ablation study framework integrates seamlessly with:
- Existing model classes in `src/models/`
- Ensemble classes in `src/ensemble/`
- Embedding generators in `src/embeddings/`
- Training configurations in `configs/config.yaml`

## Next Steps for Researchers

1. **Run baseline experiments** with full ensemble to establish baseline performance
2. **Execute leave-one-out ablations** for each model to identify most critical components
3. **Test voting strategies** to determine optimal ensemble method
4. **Hyperparameter sweep** on most important hyperparameters
5. **Analyze results** using statistical tests to determine significance
6. **Document findings** in research paper or technical report

## References

- Main configuration: `amp_prediction/configs/config.yaml`
- Model implementations: `amp_prediction/src/models/`
- Ensemble implementations: `amp_prediction/src/ensemble/`
- Training scripts: `amp_prediction/scripts/train_ensemble.py`

## Contact

For questions about ablation studies or to report issues, please open a GitHub issue.

---

**Last Updated**: 2025-01-22
**Version**: 1.0
