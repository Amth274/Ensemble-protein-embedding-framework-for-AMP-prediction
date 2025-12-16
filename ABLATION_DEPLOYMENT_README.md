# Ablation Study HPC Deployment - Quick Start

## What This Does

Automatically deploys and runs comprehensive ablation studies on the HPC cluster (`pawan@10.240.60.36`) to systematically evaluate the contribution of different components in the AMP prediction ensemble.

## Files Created

### SLURM Job Scripts
- **`slurm_ablation_comprehensive.sh`** - Single 12-hour job running all studies sequentially
- **`slurm_ablation_parallel.sh`** - 4 parallel jobs using SLURM job arrays

### Deployment Scripts
- **`deploy_and_run_ablation.ps1`** - Windows PowerShell deployment script
- **`deploy_and_run_ablation.sh`** - Linux/WSL/Git Bash deployment script
- **`submit_ablation_studies.sh`** - Job submission wrapper for HPC

### Documentation
- **`docs/ABLATION_HPC_GUIDE.md`** - Complete deployment and usage guide

## Quick Start (Choose One)

### Option 1: Windows PowerShell ⚡ RECOMMENDED
```powershell
# Open PowerShell in project directory
cd D:\Github\Ensemble-Protein-Embedding-Framework-for-AMP-Prediction-2

# Run deployment
.\deploy_and_run_ablation.ps1

# When prompted, select mode:
# - comprehensive (12h single job) - RECOMMENDED
# - parallel (4 jobs, faster)
# - both (run both modes)
# - no (deploy only, don't submit)
```

### Option 2: Git Bash / WSL
```bash
# Navigate to project
cd /d/Github/Ensemble-Protein-Embedding-Framework-for-AMP-Prediction-2

# Run deployment
./deploy_and_run_ablation.sh

# Follow prompts
```

### Option 3: Manual SSH
```bash
# 1. Sync files
rsync -avz --exclude 'venv/' --exclude '.git/' \
    ./ pawan@10.240.60.36:~/amp_prediction/

# 2. SSH and setup
ssh pawan@10.240.60.36
cd ~/amp_prediction
python3 -m venv venv
source venv/bin/activate
cd amp_prediction && pip install -e ".[dev,viz,tracking]" && cd ..

# 3. Submit jobs
./submit_ablation_studies.sh comprehensive
```

## What Gets Tested

### ✅ Embedding Ablation
- ESM-150M vs ESM-650M vs ESM-3B models
- Mean vs Max vs CLS pooling strategies
- Per-residue vs sequence-level embeddings

### ✅ Model Architecture Ablation
- Full ensemble (all 6 models)
- Leave-one-out analysis (remove each model)
- Minimal ensembles (CNN+LSTM, recurrent-only)

### ✅ Ensemble Strategy Ablation
- Soft vs Hard vs Weighted voting
- Classification thresholds (0.5 to 0.9)
- Weight computation methods

### ✅ Training Hyperparameter Ablation
- Learning rates: 1e-4, 3e-4, 5e-4, 1e-3
- Dropout rates: 0.1 to 0.5
- Batch sizes: 32, 64, 128
- Optimizers: Adam, AdamW, SGD

### ✅ Multi-Seed Validation
- 5 different random seeds
- Statistical robustness testing

## Monitoring

```bash
# Check job status
ssh pawan@10.240.60.36 'squeue -u pawan'

# View real-time logs
ssh pawan@10.240.60.36 'tail -f ~/amp_prediction/logs/ablation_study_*.out'

# Check results
ssh pawan@10.240.60.36 'ls -lh ~/amp_prediction/results/ablation/'
```

## Download Results

```bash
# After completion (~12 hours)
rsync -avz pawan@10.240.60.36:~/amp_prediction/results/ablation/ \
    ./results/ablation/
```

## Expected Timeline

### Comprehensive Mode (RECOMMENDED)
- **Deployment**: ~5 minutes
- **Queue wait**: Depends on cluster load
- **Execution**: ~12 hours
- **Total**: ~12-16 hours from start to results

### Parallel Mode
- **Deployment**: ~5 minutes
- **Queue wait**: Depends on cluster load (4 GPUs needed)
- **Execution**: ~4 hours (wall time)
- **Total**: ~4-8 hours from start to results

## Resource Usage

- **GPU**: H100 (12 GPU-hours for comprehensive, 16 GPU-hours for parallel)
- **RAM**: 256GB per job
- **CPUs**: 32 cores per job
- **Storage**: ~50GB for embeddings and results

## Troubleshooting

### Can't connect to HPC
```bash
# Test connection
ssh pawan@10.240.60.36 echo "test"

# Check SSH keys
ls ~/.ssh/
```

### rsync not found (Windows)
- Use the PowerShell script instead: `.\deploy_and_run_ablation.ps1`
- Or install Git Bash from https://git-scm.com/downloads

### Job fails
```bash
# Check error log
ssh pawan@10.240.60.36 'cat ~/amp_prediction/logs/ablation_study_*.err'

# Common fixes:
# 1. Rerun setup: ssh pawan@10.240.60.36 'cd ~/amp_prediction && source venv/bin/activate && pip install -e amp_prediction'
# 2. Check GPU quota: ssh pawan@10.240.60.36 'sinfo -p gpu-h100'
```

## After Completion

1. **Download results**: `rsync -avz pawan@10.240.60.36:~/amp_prediction/results/ablation/ ./results/ablation/`
2. **Analyze**: Review JSON files in `results/ablation/`
3. **Visualize**: Use scripts in `amp_prediction/scripts/ablation/`
4. **Document**: Update findings in project documentation

## Need Help?

See **`docs/ABLATION_HPC_GUIDE.md`** for comprehensive documentation including:
- Detailed troubleshooting
- Configuration options
- Results interpretation
- Best practices

## Project Context

This is part of the **AMP Prediction Ensemble Framework** that uses ESM-650M embeddings and 6 deep learning models to predict antimicrobial peptides with 99.16% precision and 0.9939 ROC-AUC.

The ablation studies help identify which components are critical vs redundant for achieving this high performance.
