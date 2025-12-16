# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Antimicrobial Peptide (AMP) prediction framework using ESM-650M protein language model embeddings and ensemble deep learning. The system achieves 99.16% precision and 0.9939 ROC-AUC for classification by combining 6 different neural network architectures.

**Key Innovation**: Uses ESM-650M embeddings (1280-dimensional) instead of traditional one-hot encoding, providing rich protein representations that capture evolutionary and structural information.

## Repository Structure

```
.
├── amp_prediction/              # Main package
│   ├── src/                     # Core source code
│   │   ├── models/              # Neural network architectures (CNN, LSTM, GRU, Transformer, etc.)
│   │   ├── embeddings/          # ESM embedding generation utilities
│   │   ├── ensemble/            # Ensemble strategies (voting, weighting)
│   │   ├── data/                # Data loading and preprocessing
│   │   └── utils/               # Helper utilities
│   ├── scripts/                 # Training and utility scripts
│   │   ├── generate_embeddings.py
│   │   ├── train_amp_models.py
│   │   ├── train_ensemble.py
│   │   ├── baselines/           # Baseline model training
│   │   ├── data_quality/        # Data deduplication and splitting
│   │   └── negatives/           # Hard negative generation
│   ├── app/                     # Web applications
│   │   ├── flask_app/           # Production Flask app
│   │   ├── streamlit_app/       # Alternative Streamlit interface
│   │   ├── notebooks/           # Jupyter demo notebooks
│   │   ├── examples/            # Sample data
│   │   └── utils/               # Demo utilities
│   ├── configs/                 # YAML configuration files
│   ├── data/                    # Datasets and embeddings
│   │   ├── embeddings/          # Cached ESM embeddings
│   │   ├── raw/                 # Raw FASTA files
│   │   └── *.csv                # Processed datasets
│   ├── models/                  # Saved model checkpoints
│   ├── tests/                   # Unit tests
│   ├── requirements.txt         # Dependencies
│   └── setup.py                 # Package installation
├── legacy/                      # Legacy scripts (archived)
├── docs/                        # Documentation
├── README.md                    # Main documentation
└── CLAUDE.md                    # This file
```

## Development Commands

### Installation
```bash
# Install package in development mode (from amp_prediction/ directory)
cd amp_prediction
pip install -e .

# Install with optional dependencies
pip install -e ".[dev,viz,tracking]"
pip install -e ".[dev,viz,optimization,tracking]"  # All optional dependencies
```

### Embedding Generation
```bash
# Using CLI command (if installed with `pip install -e .`)
amp-embed --config configs/config.yaml --embedding_type all

# Or run script directly
cd amp_prediction/scripts
python generate_embeddings.py --config ../configs/config.yaml --embedding_type all

# Generate specific embeddings for classification
python generate_embeddings.py --embedding_type amino_acid --task_type classification

# Generate for regression (MIC prediction)
python generate_embeddings.py --embedding_type amino_acid --task_type regression

# Specify output directory
python generate_embeddings.py --embedding_type all --output_dir ../data/embeddings
```

### Training Models
```bash
# Train individual models using the standalone script
cd amp_prediction/scripts
python train_amp_models.py \
    --train_data ../data/embeddings/train_emb_synthetic.pt \
    --test_data ../data/embeddings/test_emb_synthetic.pt \
    --epochs 20 --batch_size 64 --output_dir ../models

# Train ensemble (classification or regression)
python train_ensemble.py --config ../configs/config.yaml --task classification
python train_ensemble.py --config ../configs/config.yaml --task regression

# Note: CLI commands (amp-train, amp-evaluate, amp-predict) are defined in setup.py
# but may require corresponding implementation in src/cli.py, scripts/train.py, etc.
```

### Testing
```bash
# Run tests (from project root)
cd amp_prediction
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest app/test_flask_app.py -v
```

### Web Applications
```bash
# Launch Flask web interface (from amp_prediction/app/)
cd amp_prediction/app
python run_flask_app.py
# Access at http://127.0.0.1:5000

# Or run Flask app directly
cd amp_prediction/app/flask_app
python app.py

# Launch Streamlit app
cd amp_prediction/app/streamlit_app
python run_app.py
```

### Code Quality
```bash
# Format code (from project root or amp_prediction/)
black src/ scripts/ tests/

# Lint code
flake8 src/ scripts/ tests/

# Type checking
mypy src/
```

### Ablation Studies
```bash
# Run all ablation studies
cd amp_prediction/scripts/ablation
python run_ablation.py --config ../../configs/ablation_config.yaml --study all

# Run specific ablation studies
python run_ablation.py --study embedding  # Embedding ablation
python run_ablation.py --study model      # Model architecture ablation
python run_ablation.py --study ensemble   # Ensemble strategy ablation
python run_ablation.py --study training   # Training hyperparameter ablation

# Run with multiple seeds for robustness
for seed in 42 123 456; do
    python run_ablation.py --study all --seed $seed --results-dir ../../results/ablation
done
```

## Architecture Overview

### Core Package Structure (`amp_prediction/src/`)

**`models/`** - Neural network architectures
- All models inherit from `BaseAMPModel` (`base.py`)
- Input shape: `[batch_size, seq_len, 1280]` (ESM embeddings)
- Output shape: `[batch_size]` (binary classification) or regression value
- 6 architectures:
  - `cnn.py`: CNN1DAMPClassifier - 1D CNN with multi-scale kernels
  - `lstm.py`: AMPBilstmClassifier, AMP_BiRNN - Bidirectional LSTM/RNN models
  - `gru.py`: GRUClassifier - Bidirectional GRU
  - `hybrid.py`: CNN_BiLSTM_Classifier - Hybrid CNN+LSTM
  - `transformer.py`: AMPTransformerClassifier - Transformer encoder
  - `logistic.py`: LogisticRegression - Simple baseline

**`embeddings/`** - ESM embedding generation
- `ESMSequenceEmbedding`: Generates sequence-level embeddings (mean pooling)
- `ESMAminoAcidEmbedding`: Generates per-amino-acid embeddings for sequence modeling
- Uses `facebook/esm2_t33_650M_UR50D` model (1280-dim embeddings)
- Supports batch processing with configurable batch sizes

**`ensemble/`** - Ensemble strategies
- `ensemble_classifier.py`: Combines multiple models for classification
- `ensemble_regressor.py`: Combines multiple models for MIC prediction
- `voting.py`: Voting strategies
  - `SoftVoting`: Averages probabilities (default for classification)
  - `HardVoting`: Majority vote
  - `WeightedVoting`: Weighted averaging (for regression)

**`data/`** - Data handling
- `dataset.py`: PyTorch datasets for embeddings
- `preprocessing.py`: Data preprocessing utilities
- Expected CSV format: `Sequence,label[,value]`

### Key Design Patterns

1. **Two-stage pipeline**:
   - Stage 1: Generate ESM embeddings (computationally expensive, done once)
   - Stage 2: Train/evaluate models on cached embeddings (fast iteration)

2. **Embedding types**:
   - **Sequence embeddings**: Mean-pooled, used for logistic regression baseline
   - **Amino acid embeddings**: Per-residue, used for all deep learning models
   - **AA matrix**: Used for EvoGradient sequence optimization

3. **Model input handling**:
   - Most models accept `[batch, seq_len, 1280]` tensors
   - Logistic model needs mean-pooled input: `x.mean(dim=1)`
   - CNN models transpose to `[batch, 1280, seq_len]` internally

4. **Ensemble configuration**:
   - Soft voting with threshold τ=0.78 for classification (optimized)
   - Weighted averaging (inverse MSE weights) for regression
   - Models can be loaded/saved independently

## Configuration System

All settings are in `amp_prediction/configs/config.yaml`:

- **`data`**: Paths, sequence length constraints, train/test split ratios
- **`embedding`**: ESM model name, device, batch size, pooling strategy
- **`models`**: Per-model hyperparameters (hidden dims, dropout, layers)
- **`training`**: Learning rate, epochs, early stopping, optimizer settings
- **`ensemble`**: Voting strategy, threshold, weighting method
- **`paths`**: Output directories for models, embeddings, results, logs

## Important Implementation Details

### Embedding Generation
- ESM-650M requires significant GPU memory (16GB+ recommended)
- Sequences are padded/truncated to `max_length=100` (default)
- Embeddings are saved as PyTorch `.pt` files with metadata
- For classification: generate with `task_type=classification`
- For regression: include `value_column` in config for MIC values
- **For HPC/SLURM users**: Set GPU time limit to 12 hours for large-scale embedding generation jobs

### Model Training
- All models use BCEWithLogitsLoss for classification
- MSE loss for regression tasks
- Adam optimizer with weight decay (1e-5)
- Cosine annealing learning rate scheduler
- Early stopping based on validation AUC/MSE (patience=10)
- Best model state is saved based on validation performance

### Ensemble Predictions
- `predict_single_model()`: Get predictions from one model
- `predict_all_models()`: Get predictions from all models
- `predict_ensemble()`: Combine predictions using voting strategy
- Threshold for classification is configurable (default 0.5, optimized 0.78)

### Data Requirements
- **Classification**: CSV with `Sequence,label` columns (label: 0=non-AMP, 1=AMP)
- **Regression**: CSV with `Sequence,label,value` columns (value: MIC in μM)
- Dataset sources: APD3, LAMP, LAMP2, BAGEL4, dbAMP (AMPs) + UniProt (non-AMPs)
- Total ~50K sequences in full dataset (not included in repo)

## Recent Additions and Updates

### Ablation Study Framework (October 2025)
- Comprehensive ablation study capabilities added in `scripts/ablation/`
- Three detailed documentation files in `docs/`:
  - `ABLATION_GUIDE.md`: Complete methodology and best practices
  - `ABLATION_COMPONENTS_SUMMARY.md`: All identifiable components
  - `ABLATION_QUICK_REFERENCE.md`: Quick start guide
- Configuration file: `configs/ablation_config.yaml`
- Supports systematic testing of embeddings, models, ensemble strategies, and hyperparameters

### Repository Restructuring
- Legacy scripts moved to `legacy/` directory
- Main package reorganized under `amp_prediction/` with proper Python package structure
- Documentation consolidated in `docs/` directory
- See `docs/RESTRUCTURING_SUMMARY.md` and `docs/RESTRUCTURING_CHANGES.md` for details

### Development Roadmap
- 22 pre-written GitHub issues available in `docs/GITHUB_ISSUES.md`
- Organized into 4 phases: Testing & Quality, Documentation, Production Features, Advanced Features
- Project setup guide in `docs/PROJECT_SETUP.md`

## Special Considerations

### Working with Models
- Model architectures are defined in `src/models/` (package structure)
- Standalone training script `scripts/train_amp_models.py` contains inline model definitions
- The `base.py` provides common functionality: `save_model()`, `load_model()`, `get_model_info()`

### GPU/CPU Handling
- Device selection: Use `device="auto"` in config for automatic GPU detection
- Embedding generation requires GPU for reasonable speed (~16GB VRAM)
- Model training can run on CPU but is much slower
- Inference is fast enough on CPU for small batches

### Legacy Files
- `legacy/` directory contains archived scripts: `ensemble_cls.py`, `ensemble_reg.py`, `aa_embed.py`, `esm650m_embeddings.py`
- These are superseded by the refactored code in `amp_prediction/`
- Use package CLI commands (`amp-embed`, `amp-train`, etc.) instead

### Flask App
- Located in `amp_prediction/app/flask_app/`
- Uses **mock models** for demonstration (not real trained models)
- For production: Replace `DemoPredictor` in `app/utils/demo_utils.py` with real ensemble
- Provides REST API endpoints: `/api/predict`, `/api/batch`, `/api/health`

## Performance Benchmarks

### Classification (Ensemble)
- Accuracy: 92.50%
- Precision: 99.16%
- Recall: 85.45%
- F1-Score: 91.80%
- ROC-AUC: 99.39%

### Regression (Ensemble, MIC Prediction)
- Pearson R: 0.834
- MSE: 0.3631
- RMSE: 0.6026
- R²: 0.8371

Individual model performance is detailed in README.md tables.

## Ablation Study Framework

The repository includes a comprehensive ablation study framework to systematically evaluate component contributions.

### Overview
- **Location**: `amp_prediction/scripts/ablation/`
- **Configuration**: `amp_prediction/configs/ablation_config.yaml`
- **Documentation**: `docs/ABLATION_GUIDE.md`, `docs/ABLATION_COMPONENTS_SUMMARY.md`, `docs/ABLATION_QUICK_REFERENCE.md`

### Key Components Tested
1. **Protein Embeddings**: ESM model variants (150M, 650M, 3B), pooling strategies (mean, max, CLS)
2. **Model Architectures**: Individual models (CNN, BiLSTM, GRU, LSTM, BiCNN, Transformer), ensemble combinations
3. **Ensemble Strategies**: Voting methods (soft, hard, weighted, adaptive), classification thresholds (0.5-0.9)
4. **Training Hyperparameters**: Learning rates, dropout rates, batch sizes, optimizers, schedulers

### Running Ablations
```bash
cd amp_prediction/scripts/ablation
python run_ablation.py --config ../../configs/ablation_config.yaml --study all --seed 42
```

### Results Interpretation
- **Performance delta < -5%**: Component is critical
- **-5% ≤ delta < -2%**: Component is important
- **-2% ≤ delta < 0%**: Component has minor impact
- **delta ≥ 0%**: Component may be redundant

See `docs/ABLATION_GUIDE.md` for detailed methodology, best practices, and visualization examples.

## Common Tasks

### Adding a New Model Architecture
1. Create model class in `src/models/` inheriting from `BaseAMPModel`
2. Implement `forward()` method with correct input/output shapes
3. Add model config to `configs/config.yaml` under `models:`
4. Update ensemble initialization to include new model
5. Test with: `pytest tests/test_models.py -v`

### Modifying Ensemble Strategy
1. Edit voting logic in `src/ensemble/voting.py`
2. Update `EnsembleClassifier` or `EnsembleRegressor` to use new strategy
3. Add new strategy to config options in `configs/config.yaml`
4. Test with `amp-train --config configs/config.yaml`

### Generating Embeddings for Custom Dataset
1. Prepare CSV with `Sequence,label[,value]` columns
2. Update `data.train_path` and `data.test_path` in config
3. Run `amp-embed --config configs/config.yaml --embedding_type amino_acid`
4. Embeddings saved to `data/embeddings/` by default

### Evaluating Trained Models
1. Load models from checkpoint directory
2. Use `EnsembleClassifier.evaluate_ensemble()` or `EnsembleRegressor.evaluate()`
3. Metrics include per-model and ensemble performance
4. Results can be saved to JSON or displayed in console

### Running Ablation Studies
1. Review components to ablate in `docs/ABLATION_COMPONENTS_SUMMARY.md`
2. Configure ablation settings in `configs/ablation_config.yaml`
3. Run specific study: `python scripts/ablation/run_ablation.py --study [embedding|model|ensemble|training]`
4. Analyze results in `results/ablation/` directory
5. See `docs/ABLATION_GUIDE.md` for comprehensive guidance

## File Naming Conventions

- Model checkpoints: `{model_name}_model.pt` (e.g., `CNN_model.pt`)
- Embeddings: `{split}_emb_{type}.pt` (e.g., `train_emb_synthetic.pt`)
- Configs: `config.yaml` (main), `config_{variant}.yaml` (variants)
- Results: `training_results.json`, `evaluation_metrics.json`

## Dependencies

**Core**: PyTorch ≥1.12, transformers ≥4.20, scikit-learn ≥1.1, pandas, numpy, pyyaml, tqdm

**Optional**:
- Visualization: matplotlib, seaborn
- Development: pytest, black, flake8, mypy
- Optimization: optuna (for hyperparameter tuning)
- Tracking: wandb (for experiment tracking)

**Requirements**:
- Python ≥3.8, <3.11 (for full compatibility)
- CUDA-capable GPU recommended (16GB+ VRAM for embedding generation)
- 10GB+ disk space for models and embeddings

## Troubleshooting

### Import Errors
- Ensure package is installed: `pip install -e .` from `amp_prediction/`
- Check Python path includes `amp_prediction/src/`

### Out of Memory (GPU)
- Reduce embedding batch size in config: `embedding.batch_size: 8`
- Reduce training batch size: `training.batch_size: 32`
- Use gradient accumulation for effective larger batches

### Model Loading Errors
- Verify model architecture matches checkpoint
- Check device compatibility (CPU vs CUDA)
- Use `map_location='cpu'` when loading on CPU-only machine

### Slow Training
- Verify GPU is being used: `torch.cuda.is_available()`
- Check data is on GPU: `x.device`
- Consider using mixed precision training (FP16)

### Ablation Study Failures
- Ensure embeddings are pre-computed before running ablations
- Check that all model configurations are valid in `configs/ablation_config.yaml`
- Run with single seed first to debug, then multiple seeds for robustness
- Review error logs in `results/ablation/` directory
