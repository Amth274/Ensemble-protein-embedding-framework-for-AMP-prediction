# Repository Restructuring Summary

**Date**: October 10, 2025

## Overview

The repository has been restructured for better organization, maintainability, and clarity. This document summarizes the changes made.

## Changes Made

### 1. Legacy Files Archived

**Created**: `legacy/` directory

**Moved files**:
- `aa_embed.py` → `legacy/aa_embed.py`
- `ensemble_cls.py` → `legacy/ensemble_cls.py`
- `ensemble_reg.py` → `legacy/ensemble_reg.py`
- `esm650m_embeddings.py` → `legacy/esm650m_embeddings.py`

**Reason**: These are superseded by the refactored code in `amp_prediction/`. Archived for reference but not for active use.

### 2. Documentation Organized

**Created**: `docs/` directory

**Moved files**:
- `EXECUTION_RESULTS.md` → `docs/EXECUTION_RESULTS.md`
- `IMPLEMENTATION_GUIDE.md` → `docs/IMPLEMENTATION_GUIDE.md`

**Kept in root**:
- `README.md` (main project documentation)
- `CLAUDE.md` (AI assistant guidance)

### 3. Empty Directories Removed

**Removed from `amp_prediction/scripts/`**:
- `ablation/` (empty)
- `analysis/` (empty)
- `external/` (empty)
- `regression/` (empty)
- `sanity/` (empty)
- `splitting/` (empty)
- `uncertainty/` (empty)

**Removed from `amp_prediction/data/`**:
- `metadata/` (empty)
- `negatives/` (empty)
- `splits/` (empty)

**Removed from `amp_prediction/`**:
- `reproducibility/` (empty)

### 4. App Directory Simplified

**Removed**: `amp_prediction/app/static/` (duplicate)

**Action**: Moved `style.css` to `amp_prediction/app/flask_app/static/`

**Result**: Consolidated static assets in one location

### 5. Enhanced .gitignore

Created comprehensive `.gitignore` with proper handling of:
- Python artifacts (`__pycache__`, `*.pyc`, etc.)
- Virtual environments
- IDE files
- PyTorch model files (with exceptions for sample embeddings)
- Data files (with exceptions for configs and examples)
- Logs and results
- Coverage reports
- OS-specific files

### 6. Updated CLAUDE.md

- Added clear repository structure diagram
- Updated file paths to reflect new organization
- Added troubleshooting section
- Clarified legacy vs. current code locations
- Enhanced development commands with correct paths

## New Directory Structure

```
.
├── amp_prediction/              # Main package
│   ├── src/                     # Core source code
│   │   ├── models/              # Neural network architectures
│   │   ├── embeddings/          # ESM embedding generation
│   │   ├── ensemble/            # Ensemble strategies
│   │   ├── data/                # Data loading and preprocessing
│   │   └── utils/               # Helper utilities
│   ├── scripts/                 # Training and utility scripts
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
│   │   └── raw/                 # Raw FASTA files
│   ├── models/                  # Saved model checkpoints
│   ├── tests/                   # Unit tests
│   ├── requirements.txt         # Dependencies
│   └── setup.py                 # Package installation
├── legacy/                      # Archived legacy scripts
├── docs/                        # Documentation files
├── README.md                    # Main project documentation
└── CLAUDE.md                    # AI assistant guidance
```

## Benefits of Restructuring

### 1. **Clarity**
- Clear separation between active code (`amp_prediction/`) and legacy code (`legacy/`)
- Documentation centralized in `docs/`
- No confusion about which files to use

### 2. **Maintainability**
- Removed empty directories that served no purpose
- Consolidated duplicate static assets
- Proper .gitignore prevents accidental commits of generated files

### 3. **Developer Experience**
- Easier to navigate repository
- Clear entry points for different tasks
- Updated documentation reflects actual structure

### 4. **Best Practices**
- Follows Python package conventions
- Separates source code, tests, documentation, and data
- Uses proper configuration management

## Migration Guide

### For Developers

**Before**:
```bash
python aa_embed.py --input data.csv
python ensemble_cls.py --model cnn
```

**After**:
```bash
cd amp_prediction
amp-embed --config configs/config.yaml --embedding_type all
amp-train --config configs/config.yaml --task classification
```

### Import Paths

**No changes needed** - all imports in `amp_prediction/` use relative paths within the package.

### Configuration

**No changes needed** - config paths are relative to `amp_prediction/` directory.

### Data Paths

**Update if using absolute paths** - use paths relative to `amp_prediction/`:
- `data/train.csv` (not `../data/train.csv`)
- `data/embeddings/train_emb.pt`

## Verification

Run these commands to verify the restructuring:

```bash
# Check structure
find . -maxdepth 2 -type d ! -path "./.git*" | sort

# Verify package installation
cd amp_prediction
pip install -e .
python -c "import src.models; print('OK')"

# Run tests
pytest tests/ -v
```

## Next Steps

1. **Update any external documentation** that references old paths
2. **Update CI/CD pipelines** if they reference moved files
3. **Notify team members** of the new structure
4. **Archive old branches** that may reference legacy structure

## Rollback (if needed)

If issues arise, the restructuring can be reverted:

```bash
# Move legacy files back
mv legacy/*.py .

# Move docs back
mv docs/*.md .

# Remove new directories
rm -rf legacy docs
```

However, this should not be necessary as the restructuring is non-breaking for the core package functionality.

## Contact

For questions about the restructuring, please refer to:
- `CLAUDE.md` for updated development guidance
- `README.md` for project documentation
- Open an issue on GitHub for specific concerns
