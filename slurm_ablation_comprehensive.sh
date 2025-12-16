#!/usr/bin/bash
#SBATCH --job-name=amp_ablation
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/ablation_study_%j.out
#SBATCH --error=logs/ablation_study_%j.err

# ==============================================================================
# Comprehensive Ablation Study for AMP Prediction Ensemble
# ==============================================================================
# This script runs systematic ablation studies to evaluate component contributions
# Tests: Embeddings, Models, Ensemble Strategies, Training Hyperparameters
# ==============================================================================

echo "=========================================="
echo "AMP PREDICTION ABLATION STUDY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: ${SLURM_MEM_PER_NODE}M"
echo "GPU: H100"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Environment Setup
cd ~/amp_prediction || { echo "Error: Project directory not found"; exit 1; }

# Create logs directory if it doesn't exist
mkdir -p logs
mkdir -p results/ablation

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d "../venv" ]; then
    echo "Activating virtual environment from parent directory..."
    source ../venv/bin/activate
else
    echo "Warning: No virtual environment found. Using system Python."
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

# Install any missing dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade torch transformers scikit-learn pyyaml tqdm matplotlib seaborn
echo ""

# ==============================================================================
# PHASE 1: EMBEDDING ABLATION
# ==============================================================================
echo "=========================================="
echo "PHASE 1: EMBEDDING ABLATION"
echo "=========================================="
echo "Testing different ESM model variants and pooling strategies"
echo ""

# Run embedding ablation study
python amp_prediction/scripts/ablation/run_ablation.py \
    --config amp_prediction/configs/ablation_config.yaml \
    --study embedding \
    --seed 42 \
    --results-dir results/ablation/embedding \
    --verbose

EMBEDDING_STATUS=$?
if [ $EMBEDDING_STATUS -eq 0 ]; then
    echo "✓ Embedding ablation completed successfully"
else
    echo "✗ Embedding ablation failed with exit code $EMBEDDING_STATUS"
fi
echo ""

# ==============================================================================
# PHASE 2: MODEL ARCHITECTURE ABLATION
# ==============================================================================
echo "=========================================="
echo "PHASE 2: MODEL ARCHITECTURE ABLATION"
echo "=========================================="
echo "Testing individual models and ensemble combinations"
echo ""

python amp_prediction/scripts/ablation/run_ablation.py \
    --config amp_prediction/configs/ablation_config.yaml \
    --study model \
    --seed 42 \
    --results-dir results/ablation/model \
    --verbose

MODEL_STATUS=$?
if [ $MODEL_STATUS -eq 0 ]; then
    echo "✓ Model architecture ablation completed successfully"
else
    echo "✗ Model architecture ablation failed with exit code $MODEL_STATUS"
fi
echo ""

# ==============================================================================
# PHASE 3: ENSEMBLE STRATEGY ABLATION
# ==============================================================================
echo "=========================================="
echo "PHASE 3: ENSEMBLE STRATEGY ABLATION"
echo "=========================================="
echo "Testing voting strategies and classification thresholds"
echo ""

python amp_prediction/scripts/ablation/run_ablation.py \
    --config amp_prediction/configs/ablation_config.yaml \
    --study ensemble \
    --seed 42 \
    --results-dir results/ablation/ensemble \
    --verbose

ENSEMBLE_STATUS=$?
if [ $ENSEMBLE_STATUS -eq 0 ]; then
    echo "✓ Ensemble strategy ablation completed successfully"
else
    echo "✗ Ensemble strategy ablation failed with exit code $ENSEMBLE_STATUS"
fi
echo ""

# ==============================================================================
# PHASE 4: TRAINING HYPERPARAMETER ABLATION
# ==============================================================================
echo "=========================================="
echo "PHASE 4: TRAINING HYPERPARAMETER ABLATION"
echo "=========================================="
echo "Testing learning rates, dropout, batch sizes, optimizers"
echo ""

python amp_prediction/scripts/ablation/run_ablation.py \
    --config amp_prediction/configs/ablation_config.yaml \
    --study training \
    --seed 42 \
    --results-dir results/ablation/training \
    --verbose

TRAINING_STATUS=$?
if [ $TRAINING_STATUS -eq 0 ]; then
    echo "✓ Training hyperparameter ablation completed successfully"
else
    echo "✗ Training hyperparameter ablation failed with exit code $TRAINING_STATUS"
fi
echo ""

# ==============================================================================
# PHASE 5: MULTI-SEED ROBUSTNESS TESTING
# ==============================================================================
echo "=========================================="
echo "PHASE 5: MULTI-SEED ROBUSTNESS TESTING"
echo "=========================================="
echo "Running critical ablations with multiple seeds for robustness"
echo ""

for seed in 123 456 789 1011; do
    echo "Running with seed $seed..."

    python amp_prediction/scripts/ablation/run_ablation.py \
        --config amp_prediction/configs/ablation_config.yaml \
        --study all \
        --seed $seed \
        --results-dir results/ablation/multi_seed/seed_$seed \
        --verbose

    SEED_STATUS=$?
    if [ $SEED_STATUS -eq 0 ]; then
        echo "✓ Seed $seed completed successfully"
    else
        echo "✗ Seed $seed failed with exit code $SEED_STATUS"
    fi
    echo ""
done

# ==============================================================================
# RESULTS AGGREGATION AND VISUALIZATION
# ==============================================================================
echo "=========================================="
echo "RESULTS AGGREGATION AND VISUALIZATION"
echo "=========================================="

python -c "
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

print('Aggregating ablation results...')

results_dir = Path('results/ablation')
all_results = {}

# Collect results from all studies
for study_type in ['embedding', 'model', 'ensemble', 'training']:
    study_dir = results_dir / study_type
    if study_dir.exists():
        for result_file in study_dir.glob('*.json'):
            with open(result_file) as f:
                data = json.load(f)
                all_results[study_type] = data

# Create summary report
summary = {
    'timestamp': '$(date)',
    'job_id': '$SLURM_JOB_ID',
    'total_studies': len(all_results),
    'studies_completed': {
        'embedding': $EMBEDDING_STATUS == 0,
        'model': $MODEL_STATUS == 0,
        'ensemble': $ENSEMBLE_STATUS == 0,
        'training': $TRAINING_STATUS == 0
    },
    'results_summary': all_results
}

# Save summary
summary_file = results_dir / 'ablation_summary_${SLURM_JOB_ID}.json'
with open(summary_file, 'w') as f:
    json.dumps(summary, f, indent=2)

print(f'Summary saved to: {summary_file}')
print(f'Total studies run: {len(all_results)}')

# Print results table
print('\n========================================')
print('ABLATION STUDY RESULTS SUMMARY')
print('========================================')
for study, results in all_results.items():
    print(f'\n{study.upper()} ABLATION:')
    if isinstance(results, dict):
        for key, value in results.items():
            if isinstance(value, dict) and 'roc_auc' in value:
                print(f'  {key}: ROC-AUC = {value[\"roc_auc\"]:.4f}')
            elif isinstance(value, (int, float)):
                print(f'  {key}: {value:.4f}')
print('========================================')
"

echo ""
echo "=========================================="
echo "ABLATION STUDY COMPLETE"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo ""

# Print summary of all phases
echo "PHASE COMPLETION STATUS:"
echo "  Embedding Ablation: $([ $EMBEDDING_STATUS -eq 0 ] && echo '✓ PASSED' || echo '✗ FAILED')"
echo "  Model Ablation: $([ $MODEL_STATUS -eq 0 ] && echo '✓ PASSED' || echo '✗ FAILED')"
echo "  Ensemble Ablation: $([ $ENSEMBLE_STATUS -eq 0 ] && echo '✓ PASSED' || echo '✗ FAILED')"
echo "  Training Ablation: $([ $TRAINING_STATUS -eq 0 ] && echo '✓ PASSED' || echo '✗ FAILED')"
echo ""

# List output files
echo "Results saved to:"
ls -lh results/ablation/

echo ""
echo "Log files:"
echo "  Output: logs/ablation_study_${SLURM_JOB_ID}.out"
echo "  Error:  logs/ablation_study_${SLURM_JOB_ID}.err"
echo ""

# Copy results to home directory for easy access
echo "Copying results to home directory..."
cp -r results/ablation ~/ablation_results_${SLURM_JOB_ID}
echo "Results copied to: ~/ablation_results_${SLURM_JOB_ID}"
echo ""

# Exit with overall status (fail if any phase failed)
OVERALL_STATUS=0
if [ $EMBEDDING_STATUS -ne 0 ] || [ $MODEL_STATUS -ne 0 ] || [ $ENSEMBLE_STATUS -ne 0 ] || [ $TRAINING_STATUS -ne 0 ]; then
    OVERALL_STATUS=1
fi

exit $OVERALL_STATUS
