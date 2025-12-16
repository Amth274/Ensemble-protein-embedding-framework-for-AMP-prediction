#!/usr/bin/bash
#SBATCH --job-name=amp_retrain
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/retrain_%j.out
#SBATCH --error=logs/retrain_%j.err

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

# Run retraining script
echo ""
echo "========================================="
echo "RETRAINING ALL 6 MODELS"
echo "========================================="
echo ""

python3 amp_prediction/scripts/retrain_all_models.py \
    --train_data amp_prediction/data/embeddings/train_emb_synthetic.pt \
    --test_data amp_prediction/data/embeddings/test_emb_synthetic.pt \
    --output_dir amp_prediction/models \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.001 \
    --device cuda \
    --models all

echo ""
echo "Job finished at $(date)"
