# HPC Ablation Study Deployment Guide

This guide explains how to deploy and run comprehensive ablation studies on the HPC cluster.

## Overview

The ablation study framework tests the contribution of individual components by systematically removing or modifying them. This HPC deployment automates the process of running all ablation studies on the SLURM cluster.

## Available Scripts

### 1. SLURM Job Scripts

#### `slurm_ablation_comprehensive.sh`
- **Purpose**: Runs all ablation studies sequentially in a single 12-hour job
- **Configuration**: H100 GPU, 256GB RAM, 32 CPUs
- **Studies**: Embedding → Model → Ensemble → Training → Multi-seed validation
- **Use when**: You want a complete, systematic study with guaranteed order

#### `slurm_ablation_parallel.sh`
- **Purpose**: Runs ablation studies in parallel using SLURM job arrays
- **Configuration**: 4 separate jobs (H100 GPU, 128GB RAM, 16 CPUs each)
- **Studies**: All 4 studies run simultaneously
- **Use when**: You want faster results and have quota for multiple GPUs

### 2. Submission Scripts

#### `submit_ablation_studies.sh`
- **Purpose**: Submit ablation jobs to SLURM
- **Usage**:
  ```bash
  ./submit_ablation_studies.sh [comprehensive|parallel|both]
  ```
- **Options**:
  - `comprehensive`: Single sequential job (12 hours)
  - `parallel`: 4 parallel jobs (faster overall)
  - `both`: Submit both for comparison

### 3. Deployment Scripts

#### `deploy_and_run_ablation.sh` (Bash/Linux/WSL)
- **Purpose**: Deploy project to HPC and submit jobs
- **Features**:
  - Syncs code using rsync
  - Sets up Python environment
  - Installs dependencies
  - Submits jobs interactively

#### `deploy_and_run_ablation.ps1` (PowerShell/Windows)
- **Purpose**: Same as above but for Windows PowerShell
- **Features**: Same as Bash version with Windows compatibility

## Quick Start

### Option 1: Windows PowerShell

```powershell
# Navigate to project directory
cd D:\Github\Ensemble-Protein-Embedding-Framework-for-AMP-Prediction-2

# Run deployment script
.\deploy_and_run_ablation.ps1

# Follow prompts to select submission mode
```

### Option 2: WSL/Linux/Git Bash

```bash
# Navigate to project directory
cd /d/Github/Ensemble-Protein-Embedding-Framework-for-AMP-Prediction-2

# Make scripts executable
chmod +x deploy_and_run_ablation.sh submit_ablation_studies.sh

# Run deployment
./deploy_and_run_ablation.sh

# Follow prompts
```

### Option 3: Manual Deployment

```bash
# 1. Sync files to HPC
rsync -avz --exclude 'venv/' --exclude '.git/' \
    ./ pawan@10.240.60.36:~/amp_prediction/

# 2. SSH to HPC
ssh pawan@10.240.60.36

# 3. Setup environment
cd ~/amp_prediction
python3 -m venv venv
source venv/bin/activate
cd amp_prediction
pip install -e ".[dev,viz,tracking]"
cd ..

# 4. Submit jobs
./submit_ablation_studies.sh comprehensive
```

## Ablation Studies Included

### Phase 1: Embedding Ablation
Tests different ESM model variants and pooling strategies:
- ESM-150M vs ESM-650M vs ESM-3B
- Mean pooling vs Max pooling vs CLS token
- Per-residue vs sequence-level embeddings

### Phase 2: Model Architecture Ablation
Tests individual models and ensemble combinations:
- Full ensemble (6 models)
- Remove one model at a time (6 experiments)
- Minimal ensembles (CNN+LSTM, recurrent-only, etc.)

### Phase 3: Ensemble Strategy Ablation
Tests voting strategies and thresholds:
- Soft voting vs Hard voting vs Weighted voting
- Classification thresholds: 0.5, 0.6, 0.7, 0.78, 0.8, 0.9
- Weight computation methods for regression

### Phase 4: Training Hyperparameter Ablation
Tests training configurations:
- Learning rates: 1e-4, 3e-4, 5e-4, 1e-3
- Dropout rates: 0.1, 0.2, 0.3, 0.4, 0.5
- Batch sizes: 32, 64, 128
- Optimizers: Adam, AdamW, SGD
- Schedulers: Cosine, Step, Plateau, None

### Phase 5: Multi-Seed Robustness
Repeats critical experiments with different seeds:
- Seeds: 42, 123, 456, 789, 1011
- Provides mean and standard deviation for all metrics

## Monitoring Jobs

### Check job status
```bash
# Via SSH
ssh pawan@10.240.60.36 'squeue -u pawan'

# Or after logging in
squeue -u pawan
```

### View real-time logs
```bash
# Comprehensive job
ssh pawan@10.240.60.36 'tail -f ~/amp_prediction/logs/ablation_study_*.out'

# Parallel jobs
ssh pawan@10.240.60.36 'tail -f ~/amp_prediction/logs/ablation_parallel_*.out'
```

### Check results
```bash
ssh pawan@10.240.60.36 'ls -lh ~/amp_prediction/results/ablation/'
```

## Downloading Results

### After job completion

```bash
# Download all results
rsync -avz pawan@10.240.60.36:~/amp_prediction/results/ablation/ \
    ./results/ablation/

# Download specific study
rsync -avz pawan@10.240.60.36:~/amp_prediction/results/ablation/embedding/ \
    ./results/ablation/embedding/
```

### Results structure
```
results/ablation/
├── embedding/
│   ├── esm150m_results.json
│   ├── esm650m_results.json
│   └── esm3b_results.json
├── model/
│   ├── full_ensemble_results.json
│   ├── without_cnn_results.json
│   └── ...
├── ensemble/
│   ├── soft_voting_results.json
│   ├── hard_voting_results.json
│   └── threshold_*.json
├── training/
│   ├── lr_1e-4_results.json
│   └── ...
├── multi_seed/
│   ├── seed_42/
│   ├── seed_123/
│   └── ...
└── ablation_summary_<job_id>.json
```

## Expected Runtime

### Comprehensive Mode (Sequential)
- **Total**: ~12 hours
- **Embedding**: ~2 hours
- **Model**: ~4 hours
- **Ensemble**: ~2 hours
- **Training**: ~3 hours
- **Multi-seed**: ~1 hour

### Parallel Mode (Job Array)
- **Total**: ~4 hours (wall time)
- All studies run simultaneously
- Requires 4 GPU allocations

## Resource Requirements

### Per Job
- **GPU**: 1x H100 (or A100)
- **RAM**: 128GB minimum, 256GB recommended
- **CPUs**: 16-32 cores
- **Storage**: ~50GB for embeddings and models

### Total for Parallel
- **GPUs**: 4x H100 simultaneously
- **RAM**: 512GB total (128GB × 4)
- **CPUs**: 64 cores total

## Troubleshooting

### SSH connection fails
```bash
# Test connection
ssh pawan@10.240.60.36 echo "Connected"

# Check SSH config
cat ~/.ssh/config
```

### rsync not found (Windows)
- Install Git Bash: https://git-scm.com/downloads
- Or use WSL: `wsl --install`
- Or use PowerShell script instead

### Job fails immediately
```bash
# Check job status
squeue -u pawan

# View error log
cat ~/amp_prediction/logs/ablation_study_<job_id>.err

# Common issues:
# - Missing dependencies: Run setup again
# - GPU not available: Check partition with sinfo
# - Out of memory: Reduce batch size in config
```

### Out of GPU memory
Edit `amp_prediction/configs/ablation_config.yaml`:
```yaml
training_ablation:
  batch_sizes:
    - 16  # Reduce from 32
    - 32  # Reduce from 64
    - 64  # Reduce from 128
```

### Job timeout
If 12 hours is not enough:
```bash
# Edit slurm_ablation_comprehensive.sh
#SBATCH --time=24:00:00  # Increase to 24 hours
```

## Best Practices

### Before Running
1. ✅ Test code locally on small dataset
2. ✅ Ensure embeddings are generated
3. ✅ Check HPC quota: `squeue -u pawan`
4. ✅ Verify GPU availability: `sinfo -p gpu-h100`

### During Execution
1. 📊 Monitor job status regularly
2. 📈 Check log files for errors
3. 💾 Verify results are being written

### After Completion
1. 📥 Download results immediately
2. 📊 Generate visualizations (see ABLATION_GUIDE.md)
3. 📝 Document findings
4. 🧹 Clean up temporary files on HPC

## Configuration

### Modify ablation parameters
Edit `amp_prediction/configs/ablation_config.yaml`:

```yaml
# Example: Add new learning rate
training_ablation:
  learning_rates:
    - 1e-4
    - 3e-4
    - 5e-4
    - 1e-3
    - 2e-3  # Add this
```

### Modify SLURM resources
Edit `slurm_ablation_comprehensive.sh`:

```bash
#SBATCH --mem=512G        # Increase memory
#SBATCH --time=24:00:00   # Increase time
#SBATCH --cpus-per-task=64  # More CPUs
```

## Related Documentation

- **ABLATION_GUIDE.md**: Comprehensive ablation methodology
- **ABLATION_COMPONENTS_SUMMARY.md**: All testable components
- **ABLATION_QUICK_REFERENCE.md**: Quick start guide
- **CLAUDE.md**: Project structure and commands

## Support

### If you encounter issues:
1. Check error logs: `logs/ablation_study_*.err`
2. Review SLURM output: `logs/ablation_study_*.out`
3. Verify configuration: `configs/ablation_config.yaml`
4. Test locally: `python amp_prediction/scripts/ablation/run_ablation.py --help`

### Common Questions

**Q: Can I run multiple ablation jobs simultaneously?**
A: Yes, use the parallel mode or submit multiple comprehensive jobs with different configs.

**Q: How much does this cost in compute credits?**
A: ~48 GPU-hours for comprehensive, ~16 GPU-hours for parallel (4 GPUs × 4 hours)

**Q: Can I cancel a running job?**
A: Yes, use `scancel <job_id>` or `scancel -u pawan` to cancel all your jobs

**Q: How do I resume a failed job?**
A: Check which phase failed in the logs, modify the script to start from that phase, and resubmit.

## Example Session

```bash
# Full workflow example
cd D:\Github\Ensemble-Protein-Embedding-Framework-for-AMP-Prediction-2

# Deploy and run
.\deploy_and_run_ablation.ps1
# Select: comprehensive

# Monitor (in another terminal)
ssh pawan@10.240.60.36
squeue -u pawan
tail -f ~/amp_prediction/logs/ablation_study_*.out

# When complete, download results
rsync -avz pawan@10.240.60.36:~/amp_prediction/results/ablation/ ./results/ablation/

# Analyze results
cd results/ablation
python ../scripts/visualize_ablation.py --results-dir .
```
