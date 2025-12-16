#!/usr/bin/bash
#SBATCH --job-name=amp_abl_complete
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_complete_%j.out
#SBATCH --error=logs/ablation_complete_%j.err

# ==============================================================================
# COMPLETE Ablation Study with Full Implementation
# ==============================================================================
# This script runs the fully implemented ablation study that:
# 1. Loads pre-trained models and evaluates them
# 2. Tests different ensemble combinations
# 3. Tests different classification thresholds
# 4. (Optional) Retrains models with different hyperparameters
# ==============================================================================

echo "=========================================="
echo "AMP PREDICTION - COMPLETE ABLATION STUDY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}M"
echo "GPU: H100"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Navigate to project directory
cd ~/amp_prediction || { echo "Error: Project directory not found"; exit 1; }

# Create necessary directories
mkdir -p logs
mkdir -p results/ablation

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found"
    exit 1
fi

# Set environment variables for optimal GPU usage
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

# Verify GPU availability
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo ""

# Verify data and models exist
echo "Verifying data and models..."
echo "Embeddings:"
ls -lh amp_prediction/data/embeddings/*.pt || { echo "Error: Embeddings not found"; exit 1; }
echo ""
echo "Models:"
ls -lh amp_prediction/models/*.pt || { echo "Error: Models not found"; exit 1; }
echo ""

# ==============================================================================
# RUN COMPLETE ABLATION STUDY
# ==============================================================================
echo "=========================================="
echo "RUNNING COMPLETE ABLATION STUDY"
echo "=========================================="
echo ""

# Run in quick mode (no hyperparameter training) or full mode
# Change --quick to enable/disable hyperparameter ablation
python amp_prediction/scripts/ablation/run_real_ablation.py \
    --config amp_prediction/configs/ablation_config.yaml \
    --data-dir amp_prediction/data/embeddings \
    --models-dir amp_prediction/models \
    --results-dir results/ablation \
    --seed 42 \
    --device auto \
    --quick

ABLATION_STATUS=$?

if [ $ABLATION_STATUS -eq 0 ]; then
    echo ""
    echo "✓ Ablation study completed successfully"
else
    echo ""
    echo "✗ Ablation study failed with exit code $ABLATION_STATUS"
    exit $ABLATION_STATUS
fi

# ==============================================================================
# GENERATE RESULTS SUMMARY
# ==============================================================================
echo ""
echo "=========================================="
echo "GENERATING RESULTS SUMMARY"
echo "=========================================="
echo ""

python -c "
import json
from pathlib import Path
import sys

results_dir = Path('results/ablation')

print('='*80)
print('ABLATION STUDY RESULTS SUMMARY')
print('='*80)

# Load model architecture results
model_file = results_dir / 'model_architecture_results.json'
if model_file.exists():
    print('\n1. MODEL ARCHITECTURE ABLATION:')
    print('-'*80)
    with open(model_file) as f:
        model_results = json.load(f)

    # Sort by F1 score
    sorted_results = sorted(
        model_results.items(),
        key=lambda x: x[1].get('f1_score', 0),
        reverse=True
    )

    for name, metrics in sorted_results:
        if 'individual' in name:
            continue
        print(f'{name:25s} | F1: {metrics.get(\"f1_score\", 0):.4f} | '
              f'AUC: {metrics.get(\"roc_auc\", 0):.4f} | '
              f'Prec: {metrics.get(\"precision\", 0):.4f} | '
              f'Rec: {metrics.get(\"recall\", 0):.4f}')

    print('\n   Individual Models:')
    for name, metrics in sorted_results:
        if 'individual' not in name:
            continue
        model_name = name.replace('individual_', '')
        print(f'   {model_name:15s} | F1: {metrics.get(\"f1_score\", 0):.4f} | '
              f'AUC: {metrics.get(\"roc_auc\", 0):.4f}')

    # Find best combination
    best = max([(k, v) for k, v in model_results.items() if 'individual' not in k],
               key=lambda x: x[1].get('f1_score', 0))

    print(f'\n   Best: {best[0]} (F1: {best[1].get(\"f1_score\", 0):.4f})')

# Load threshold results
thresh_file = results_dir / 'threshold_results.json'
if thresh_file.exists():
    print('\n2. CLASSIFICATION THRESHOLD ABLATION:')
    print('-'*80)
    with open(thresh_file) as f:
        threshold_results = json.load(f)

    for name, metrics in sorted(threshold_results.items()):
        thresh_val = metrics.get('threshold', 0)
        print(f'Threshold {thresh_val:.2f} | '
              f'Prec: {metrics.get(\"precision\", 0):.4f} | '
              f'Rec: {metrics.get(\"recall\", 0):.4f} | '
              f'F1: {metrics.get(\"f1_score\", 0):.4f}')

    # Find best threshold
    best_thresh = max(threshold_results.items(),
                     key=lambda x: x[1].get('f1_score', 0))
    print(f'\n   Best: {best_thresh[1].get(\"threshold\", 0):.2f} '
          f'(F1: {best_thresh[1].get(\"f1_score\", 0):.4f})')

# Load hyperparameter results if they exist
hyperparam_file = results_dir / 'hyperparameter_results.json'
if hyperparam_file.exists():
    print('\n3. HYPERPARAMETER ABLATION:')
    print('-'*80)
    with open(hyperparam_file) as f:
        hyperparam_results = json.load(f)

    for name, metrics in sorted(hyperparam_results.items()):
        lr = metrics.get('learning_rate', 0)
        print(f'LR {lr:.0e} | '
              f'F1: {metrics.get(\"f1_score\", 0):.4f} | '
              f'AUC: {metrics.get(\"roc_auc\", 0):.4f}')

    best_lr = max(hyperparam_results.items(),
                  key=lambda x: x[1].get('f1_score', 0))
    print(f'\n   Best: {best_lr[1].get(\"learning_rate\", 0):.0e} '
          f'(F1: {best_lr[1].get(\"f1_score\", 0):.4f})')
else:
    print('\n3. HYPERPARAMETER ABLATION:')
    print('-'*80)
    print('   Skipped (quick mode)')

print('\n' + '='*80)
print('Results files saved in: results/ablation/')
for result_file in results_dir.glob('*_results.json'):
    print(f'  - {result_file.name}')
print('='*80)
"

echo ""
echo "=========================================="
echo "ABLATION STUDY COMPLETE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo ""

# Copy results to home directory for easy access
echo "Copying results to home directory..."
cp -r results/ablation ~/ablation_results_complete_${SLURM_JOB_ID}
echo "Results copied to: ~/ablation_results_complete_${SLURM_JOB_ID}"
echo ""

# Display results location
echo "Results available at:"
echo "  HPC: /export/home/pawan/amp_prediction/results/ablation/"
echo "  Backup: ~/ablation_results_complete_${SLURM_JOB_ID}/"
echo ""
echo "Download with:"
echo "  scp -r pawan@10.240.60.36:~/ablation_results_complete_${SLURM_JOB_ID}/ ."
echo ""

exit $ABLATION_STATUS
