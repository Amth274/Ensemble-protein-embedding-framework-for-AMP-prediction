#!/usr/bin/bash
#SBATCH --job-name=amp_validation
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/validation_%j.out
#SBATCH --error=logs/validation_%j.err

echo "Job started at $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Activate virtual environment
source /export/home/pawan/amp_prediction/venv/bin/activate

# Navigate to project directory
cd /export/home/pawan/amp_prediction

# Print GPU info
echo "GPU Info:"
nvidia-smi

# Run comprehensive validation
echo ""
echo "========================================="
echo "COMPREHENSIVE VALIDATION"
echo "========================================="
echo ""

python3 amp_prediction/scripts/comprehensive_validation.py \
    --test_data amp_prediction/data/embeddings/test_emb_synthetic.pt \
    --train_data amp_prediction/data/embeddings/train_emb_synthetic.pt \
    --models_dir amp_prediction/models \
    --output_dir results/validation \
    --device cuda

echo ""
echo "Job finished at $(date)"
