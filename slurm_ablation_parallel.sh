#!/usr/bin/bash
#SBATCH --job-name=amp_abl_%A_%a
#SBATCH --partition=gpu-h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:1
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/ablation_parallel_%A_%a.out
#SBATCH --error=logs/ablation_parallel_%A_%a.err

# ==============================================================================
# Parallel Ablation Study Job Array for AMP Prediction Ensemble
# ==============================================================================
# This script runs different ablation studies in parallel using SLURM job arrays
# Array indices: 0=embedding, 1=model, 2=ensemble, 3=training
# ==============================================================================

echo "=========================================="
echo "AMP ABLATION STUDY - PARALLEL JOB ARRAY"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="
echo ""

# Environment Setup
cd ~/amp_prediction || { echo "Error: Project directory not found"; exit 1; }

mkdir -p logs
mkdir -p results/ablation

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# Define study types
declare -a STUDIES=("embedding" "model" "ensemble" "training")
STUDY_TYPE=${STUDIES[$SLURM_ARRAY_TASK_ID]}

echo "Running ablation study: $STUDY_TYPE"
echo ""

# Run the specific ablation study
python amp_prediction/scripts/ablation/run_ablation.py \
    --config amp_prediction/configs/ablation_config.yaml \
    --study $STUDY_TYPE \
    --seed 42 \
    --results-dir results/ablation/${STUDY_TYPE} \
    --verbose

STATUS=$?

echo ""
echo "=========================================="
echo "STUDY COMPLETED: $STUDY_TYPE"
echo "Status: $([ $STATUS -eq 0 ] && echo 'SUCCESS' || echo 'FAILED')"
echo "End time: $(date)"
echo "=========================================="

exit $STATUS
