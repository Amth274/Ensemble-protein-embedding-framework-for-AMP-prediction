# Enhanced Antimicrobial Peptide Prediction using ESM-650M Embeddings and Ensemble Deep Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A state-of-the-art deep learning framework for antimicrobial peptide (AMP) prediction and potency estimation, leveraging ESM-650M protein language model embeddings and ensemble learning.

## Overview

This repository implements an advanced AMP discovery pipeline that significantly outperforms existing approaches by:

- **Rich Protein Representations**: Uses ESM-650M pretrained embeddings (1280-dimensional) instead of traditional one-hot encoding
- **Diverse Model Ensemble**: Combines 6 deep learning architectures (CNN, BiLSTM, GRU, LSTM, BiCNN, Transformer)
- **Superior Performance**: Achieves 99.16% precision, 0.9939 ROC-AUC for classification, and Pearson R > 0.89 for MIC prediction
- **Production Ready**: Includes CLI tools, Flask web interface, and comprehensive configuration management

## Key Results

### Classification Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| CNN | 0.9298 | 0.9868 | 0.8686 | 0.9239 | 0.9889 |
| BiLSTM | 0.9028 | 0.9869 | 0.8129 | 0.8915 | 0.9869 |
| GRU | 0.9420 | 0.9799 | 0.9002 | 0.9384 | 0.9901 |
| LSTM | 0.9453 | 0.9864 | 0.8762 | 0.9263 | 0.9905 |
| BiCNN | 0.9530 | 0.9804 | 0.9227 | 0.9507 | 0.9918 |
| Transformer | 0.9315 | 0.9825 | 0.8762 | 0.9263 | 0.9905 |
| **Ensemble** | **0.9250** | **0.9916** | **0.8545** | **0.9180** | **0.9939** |

### Regression Performance (MIC Prediction)

| Model | Pearson R | MSE | R² |
|-------|-----------|-----|-----|
| CNN | 0.9034 | 0.4259 | 0.8090 |
| GRU | 0.9035 | 0.4127 | 0.8149 |
| LSTM | 0.8975 | 0.4433 | 0.8011 |
| BiLSTM | 0.8878 | 0.4548 | 0.7960 |
| BiCNN | 0.8970 | 0.4761 | 0.7864 |
| Transformer | 0.8794 | 0.5167 | 0.7682 |
| **Ensemble** | **0.8340** | **0.3631** | **0.8371** |

**Improvement over baseline**: 99-138% increase in Pearson R compared to original EvoGradient approach.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, 16GB+ VRAM for embedding generation)
- 10GB+ disk space for models and embeddings

### Quick Start

```bash
# Clone the repository
git clone https://github.com/PawanRamaMali/Ensemble-Protein-Embedding-Framework-for-AMP-Prediction.git
cd Ensemble-Protein-Embedding-Framework-for-AMP-Prediction

# Install the package
cd amp_prediction
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,viz,tracking]"
```

## Usage

### 1. Generate ESM Embeddings

```bash
# Generate all embedding types
amp-embed --config configs/config.yaml --embedding_type all

# Generate specific embeddings
amp-embed --embedding_type amino_acid --task_type classification
```

### 2. Train Ensemble Models

```bash
# Train classification ensemble
amp-train --config configs/config.yaml --task classification

# Train regression ensemble for MIC prediction
amp-train --config configs/config.yaml --task regression
```

### 3. Evaluate Models

```bash
# Evaluate trained ensemble
amp-evaluate --config configs/config.yaml --model_dir models/

# Evaluate with custom test data
amp-evaluate --model_dir models/ --data_path data/test_embeddings.pt
```

### 4. Make Predictions

```bash
# Predict on new sequences
amp-predict --model_dir models/ \
    --sequences "GLFDIVKKVVGALCS,GIGKFLHSAKKFGKAFVGEIMNS" \
    --output predictions.csv
```

### 5. Web Interface

```bash
# Launch Flask web application
cd amp_prediction/app
python run_flask_app.py
```

Visit `http://127.0.0.1:5000` for interactive predictions with:
- Single sequence prediction
- Batch CSV upload
- Pre-loaded AMP examples
- REST API endpoints

## Project Structure

```
Ensemble-Protein-Embedding-Framework-for-AMP-Prediction/
├── amp_prediction/              # Main package (refactored)
│   ├── src/                     # Source modules
│   │   ├── models/              # Neural network architectures
│   │   │   ├── cnn.py          # 1D CNN classifier
│   │   │   ├── lstm.py         # BiLSTM and RNN models
│   │   │   ├── gru.py          # GRU classifier
│   │   │   ├── transformer.py  # Transformer encoder
│   │   │   ├── hybrid.py       # CNN-BiLSTM hybrid
│   │   │   └── base.py         # Base model class
│   │   ├── embeddings/         # ESM embedding generation
│   │   ├── data/               # Data handling
│   │   ├── ensemble/           # Ensemble strategies
│   │   └── utils/              # Utilities
│   ├── scripts/                # Training and evaluation scripts
│   │   ├── ablation/           # Ablation study scripts ⭐
│   │   │   ├── run_ablation.py
│   │   │   └── ablation_utils.py
│   │   ├── train_ensemble.py
│   │   └── generate_embeddings.py
│   ├── configs/                # YAML configuration files
│   │   ├── config.yaml
│   │   └── ablation_config.yaml  # Ablation configuration ⭐
│   ├── app/                    # Flask web application
│   ├── tests/                  # Unit tests
│   └── requirements.txt        # Dependencies
├── docs/                       # Documentation
│   ├── ABLATION_GUIDE.md       # Comprehensive ablation guide ⭐
│   ├── ABLATION_COMPONENTS_SUMMARY.md  # Component summary ⭐
│   └── ABLATION_QUICK_REFERENCE.md     # Quick reference ⭐
├── ensemble_cls.py             # Legacy classification script
├── ensemble_reg.py             # Legacy regression script
├── aa_embed.py                 # Legacy embedding generation
└── CLAUDE.md                   # Development guidelines

```

## Model Architectures

### 1. CNN1D Classifier
- 3 convolutional layers (kernels: 3, 5, 7)
- Batch normalization and dropout
- Adaptive max pooling
- Multi-scale feature extraction

### 2. BiLSTM Classifier
- Bidirectional LSTM (256 hidden units)
- Layer normalization
- Captures long-range dependencies

### 3. GRU Classifier
- Bidirectional GRU (256 hidden units)
- Efficient training with gating mechanisms

### 4. LSTM Classifier
- Bidirectional RNN with 256 hidden units
- 2-layer architecture for deeper representations

### 5. BiCNN (Hybrid)
- CNN layer followed by BiLSTM
- Combines local and global features

### 6. Transformer Classifier
- Transformer encoder with multi-head attention
- GELU activation, mean pooling
- Captures contextual relationships

## Ensemble Strategies

### Classification (Soft Voting)
```python
p_ensemble(x) = (1/M) * Σ p_i(x)
ŷ(x) = 1 if p_ensemble(x) ≥ τ else 0
```
Where τ = 0.78 (optimized threshold)

### Regression (Weighted Averaging)
```python
ŷ_ensemble = Σ w_i * ŷ_i
where Σ w_i = 1
```
Weights computed via inverse MSE on validation set.

## Configuration

All parameters are managed via `configs/config.yaml`:

```yaml
# Embedding configuration
embedding:
  model_name: "facebook/esm2_t33_650M_UR50D"
  embedding_dim: 1280
  batch_size: 16

# Training configuration
training:
  batch_size: 64
  learning_rate: 3e-4
  num_epochs: 100
  early_stopping_patience: 10

# Ensemble configuration
ensemble:
  voting_strategy: "soft"
  threshold: 0.78
```

## Data Requirements

The system expects CSV files with the following format:

### Classification Dataset
```csv
Sequence,label
GLFDIVKKVVGALCS,1
DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQ,0
```

### Regression Dataset
```csv
Sequence,label,value
GIGKFLHSAKKFGKAFVGEIMNS,1,25.0
KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK,1,12.5
```

**Dataset Sources** (not included):
- AMPs: APD3, LAMP, LAMP2, BAGEL4, dbAMP (~24,766 sequences)
- Non-AMPs: UniProt random sample (~26,047 sequences)

## Performance Benchmarks

### Comparison with EvoGradient (Wang et al. 2025)

| Metric | Original | Ours | Improvement |
|--------|----------|------|-------------|
| Classification ROC-AUC | 0.9813 | 0.9939 | +1.3% |
| Classification Precision | 0.99 | 0.9916 | +0.16% |
| Regression Pearson R (CNN) | 0.39 | 0.9034 | +131% |
| Regression Pearson R (LSTM) | 0.40 | 0.8975 | +124% |
| Regression Pearson R (Transformer) | 0.37 | 0.8794 | +138% |

## API Reference

### REST API Endpoints

```bash
# Health check
GET /api/health

# Single prediction
POST /api/predict
{
  "sequence": "GLFDIVKKVVGALCS"
}

# Batch prediction
POST /api/batch
{
  "sequences": ["GLFDIVKKVVGALCS", "GIGKFLHSAKKFGKAFVGEIMNS"]
}
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ scripts/ tests/
flake8 src/ scripts/ tests/
```

### Type Checking
```bash
mypy src/
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{amp2025enhanced,
  title={Enhanced Antimicrobial Peptide Prediction using ESM-650M Embeddings and Ensemble Deep Learning},
  author={Aum Thaker et al.},
  journal={TBD},
  year={2025}
}
```

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- **ESM-2**: Facebook AI Research for the ESM-650M protein language model
- **EvoGradient**: Wang et al. (2025) for the baseline framework
- **AMP Databases**: APD3, LAMP, dbAMP for curated datasets

## Contact

For questions or issues, please open a GitHub issue or contact [prm@outlook.in]

## Ablation Studies

The framework includes comprehensive ablation study capabilities to identify key components and their contributions. See detailed documentation:

- **[Ablation Guide](docs/ABLATION_GUIDE.md)**: Comprehensive guide for running ablation studies
- **[Components Summary](docs/ABLATION_COMPONENTS_SUMMARY.md)**: Complete list of identified components
- **[Quick Reference](docs/ABLATION_QUICK_REFERENCE.md)**: Quick start guide and cheat sheet

### Key Components Identified for Ablation

**Protein Embeddings**: 3 ESM model variants, 3 pooling strategies
**Model Architectures**: 6 individual models (CNN, BiLSTM, GRU, LSTM, BiCNN, Transformer), 10 ensemble combinations
**Ensemble Strategies**: 4 voting methods (soft, hard, weighted, adaptive), 6 threshold values
**Training Components**: Learning rates, dropout rates, batch sizes, optimizers, schedulers

### Running Ablation Studies

```bash
# Run all ablation studies
cd amp_prediction/scripts/ablation
python run_ablation.py --config ../../configs/ablation_config.yaml --study all

# Run specific ablation study
python run_ablation.py --study model --seed 42
```

## Roadmap

- [x] Comprehensive ablation study framework
- [ ] Add hyperparameter optimization with Optuna
- [ ] Implement attention visualization
- [ ] Add SHAP-based explainability
- [ ] Support for multi-class AMP classification
- [ ] Integration with AlphaFold for structure-based features
- [ ] Docker containerization
- [ ] Pre-trained model zoo

---

**Note**: This implementation requires downloading AMP datasets from public sources and training models. Pre-trained checkpoints are not included in this repository due to size constraints.
