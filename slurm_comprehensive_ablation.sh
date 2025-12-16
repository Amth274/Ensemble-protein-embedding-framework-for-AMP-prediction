#!/usr/bin/bash
#SBATCH --job-name=amp_ablation
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/comprehensive_ablation_%j.out
#SBATCH --error=logs/comprehensive_ablation_%j.err

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

# Run comprehensive ablation study
echo ""
echo "========================================="
echo "COMPREHENSIVE ABLATION STUDY"
echo "========================================="
echo ""

python3 amp_prediction/scripts/ablation/comprehensive_ablation.py

echo ""
echo "Job finished at $(date)"
