# Repository Restructuring - Quick Reference

## What Changed?

### ✅ Improved Organization

| Before | After | Reason |
|--------|-------|--------|
| `aa_embed.py` (root) | `legacy/aa_embed.py` | Superseded by package CLI |
| `ensemble_cls.py` (root) | `legacy/ensemble_cls.py` | Refactored into `src/ensemble/` |
| `ensemble_reg.py` (root) | `legacy/ensemble_reg.py` | Refactored into `src/ensemble/` |
| `esm650m_embeddings.py` (root) | `legacy/esm650m_embeddings.py` | Refactored into `src/embeddings/` |
| `EXECUTION_RESULTS.md` (root) | `docs/EXECUTION_RESULTS.md` | Better organization |
| `IMPLEMENTATION_GUIDE.md` (root) | `docs/IMPLEMENTATION_GUIDE.md` | Better organization |
| `training_log.txt` (amp_prediction/) | `docs/training_log.txt` | Better organization |
| `VALIDATION_FRAMEWORK_PROGRESS.md` (amp_prediction/) | `docs/VALIDATION_FRAMEWORK_PROGRESS.md` | Better organization |

### 🗑️ Cleaned Up

**Removed empty directories**:
- `amp_prediction/scripts/{ablation,analysis,external,regression,sanity,splitting,uncertainty}/`
- `amp_prediction/data/{metadata,negatives,splits}/`
- `amp_prediction/reproducibility/`
- `amp_prediction/app/static/` (consolidated into `flask_app/static/`)

### 📝 Enhanced

- **`.gitignore`**: Comprehensive rules for Python, PyTorch, data files, IDEs
- **`CLAUDE.md`**: Updated with new structure, troubleshooting, and best practices

## Quick Start (Updated Commands)

### Installation
```bash
cd amp_prediction
pip install -e .
```

### Generate Embeddings
```bash
amp-embed --config configs/config.yaml --embedding_type all
```

### Train Models
```bash
cd amp_prediction/scripts
python train_amp_models.py \
    --train_data ../data/embeddings/train_emb_synthetic.pt \
    --test_data ../data/embeddings/test_emb_synthetic.pt \
    --output_dir ../models
```

### Launch Web App
```bash
cd amp_prediction/app
python run_flask_app.py
```

## New Structure (Simplified)

```
📁 Root
├── 📁 amp_prediction/          Main package
│   ├── 📁 src/                 Core source code
│   ├── 📁 scripts/             Training scripts
│   ├── 📁 app/                 Web applications
│   ├── 📁 configs/             Configuration files
│   ├── 📁 data/                Datasets
│   ├── 📁 models/              Model checkpoints
│   └── 📁 tests/               Unit tests
├── 📁 legacy/                  Archived old scripts
├── 📁 docs/                    Documentation
├── 📄 README.md                Project docs
└── 📄 CLAUDE.md                AI assistant guide
```

## Benefits

✨ **Cleaner**: No clutter in root directory
📚 **Organized**: Documentation in one place
🎯 **Clear**: Active vs. archived code separation
🚀 **Modern**: Follows Python best practices

## No Breaking Changes

- All imports within `amp_prediction/` work as before
- Configuration paths remain the same
- Package installation unchanged
- Tests run without modification

## Need Help?

- Read `CLAUDE.md` for detailed guidance
- Check `README.md` for project overview
- See `docs/RESTRUCTURING_SUMMARY.md` for full details
