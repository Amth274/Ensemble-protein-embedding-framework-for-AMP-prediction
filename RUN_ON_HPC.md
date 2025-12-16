# Run Ablation Studies on HPC - Final Instructions

## Files Have Been Transferred! ✓

The SLURM scripts and project files are now on the HPC cluster at:
`/export/home/pawan/amp_prediction`

## Run the Ablation Studies

Simply SSH to the HPC and run:

```bash
ssh pawan@10.240.60.36

cd /export/home/pawan/amp_prediction

# Fix line endings (Windows → Unix)
dos2unix *.sh 2>/dev/null || sed -i 's/\r$//' *.sh

# Make executable
chmod +x *.sh

# Setup environment and submit jobs
bash setup_and_run_on_hpc.sh
```

When prompted, choose:
- **1** for comprehensive mode (RECOMMENDED) - 12-hour single job
- **2** for parallel mode - 4 jobs running simultaneously
- **3** for both modes
- **4** to skip submission (setup only)

## Quick Submit (Skip Interactive)

If you want to submit directly without prompts:

```bash
ssh pawan@10.240.60.36 << 'EOF'
cd /export/home/pawan/amp_prediction
sed -i 's/\r$//' *.sh
chmod +x *.sh

# Create venv if needed
[ ! -d venv ] && python3 -m venv venv
source venv/bin/activate

# Install dependencies
cd amp_prediction
pip install -q -e ".[dev,viz,tracking]" || pip install -q torch transformers scikit-learn pyyaml tqdm pandas numpy matplotlib seaborn
cd ..

# Submit comprehensive ablation study
sbatch slurm_ablation_comprehensive.sh
EOF
```

## Monitor Jobs

```bash
# Check job status
ssh pawan@10.240.60.36 'squeue -u pawan'

# View real-time logs
ssh pawan@10.240.60.36 'tail -f /export/home/pawan/amp_prediction/logs/ablation_study_*.out'

# Check results directory
ssh pawan@10.240.60.36 'ls -lh /export/home/pawan/amp_prediction/results/ablation/'
```

## Download Results (After ~12 hours)

```bash
# From your local machine
scp -r pawan@10.240.60.36:/export/home/pawan/amp_prediction/results/ablation/ ./results/

# Or if in Git Bash
scp -r pawan@10.240.60.36:/export/home/pawan/amp_prediction/results/ablation/ /d/Github/Ensemble-Protein-Embedding-Framework-for-AMP-Prediction-2/results/
```

## What Will Run

### Phase 1: Embedding Ablation (~2 hours)
- ESM-150M vs ESM-650M vs ESM-3B
- Different pooling strategies

### Phase 2: Model Architecture Ablation (~4 hours)
- Full ensemble vs leave-one-out analysis
- Individual model contributions

### Phase 3: Ensemble Strategy Ablation (~2 hours)
- Voting strategies
- Classification thresholds

### Phase 4: Training Hyperparameter Ablation (~3 hours)
- Learning rates, dropout, batch sizes
- Optimizer comparisons

### Phase 5: Multi-Seed Validation (~1 hour)
- Statistical robustness testing with 5 seeds

**Total Time**: ~12 hours on H100 GPU

## Expected Output Files

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
│   └── threshold_*.json
├── training/
│   └── lr_*_results.json
└── ablation_summary_<job_id>.json
```

## Troubleshooting

### If job fails immediately:

```bash
# Check error log
ssh pawan@10.240.60.36 'cat /export/home/pawan/amp_prediction/logs/ablation_study_*.err'

# Common fix: Reinstall dependencies
ssh pawan@10.240.60.36 'cd /export/home/pawan/amp_prediction && source venv/bin/activate && pip install -e amp_prediction'
```

### Check GPU availability:

```bash
ssh pawan@10.240.60.36 'sinfo -p gpu-h100'
```

### Cancel job:

```bash
ssh pawan@10.240.60.36 'scancel -u pawan'
```
