#!/bin/bash

# ==============================================================================
# Setup and Run Ablation Studies on HPC
# ==============================================================================
# Run this script ON THE HPC CLUSTER after transferring files
# Usage: ssh pawan@10.240.60.36 'bash -s' < setup_and_run_on_hpc.sh
# Or: scp setup_and_run_on_hpc.sh pawan@10.240.60.36:~ && ssh pawan@10.240.60.36 'bash ~/setup_and_run_on_hpc.sh'
# ==============================================================================

set -e  # Exit on error

PROJECT_DIR="$HOME/amp_prediction"

echo "=========================================="
echo "AMP Ablation Study Setup on HPC"
echo "=========================================="
echo "Project directory: $PROJECT_DIR"
echo "Current directory: $(pwd)"
echo "Hostname: $(hostname)"
echo "Time: $(date)"
echo "=========================================="
echo ""

# Navigate to project directory
cd "$PROJECT_DIR" || {
    echo "Error: Project directory $PROJECT_DIR not found"
    echo "Please transfer project files first"
    exit 1
}

# Create necessary directories
echo "Creating directories..."
mkdir -p logs
mkdir -p results/ablation
mkdir -p amp_prediction/scripts/ablation
mkdir -p amp_prediction/configs
echo "✓ Directories created"
echo ""

# Setup Python virtual environment
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    echo "  Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "  Upgrading pip..."
pip install --quiet --upgrade pip setuptools wheel

echo "  Installing project dependencies..."
cd amp_prediction
pip install --quiet -e ".[dev,viz,tracking]" || {
    echo "Warning: Full install failed, trying basic install..."
    pip install torch transformers scikit-learn pyyaml tqdm pandas numpy matplotlib seaborn
}
cd ..

echo "✓ Python environment ready"
echo ""

# Verify GPU availability
echo "Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
" || echo "Warning: GPU check failed, but continuing..."
echo ""

# Make scripts executable
echo "Setting script permissions..."
chmod +x *.sh
echo "✓ Scripts are executable"
echo ""

# Show available SLURM partitions
echo "Checking SLURM configuration..."
echo "Available GPU partitions:"
sinfo -p gpu-h100 || echo "  (gpu-h100 partition info not available)"
echo ""

# Check current job queue
echo "Current job queue for user $USER:"
squeue -u $USER || echo "  No jobs running"
echo ""

# Submit ablation studies
echo "=========================================="
echo "Ready to submit ablation studies!"
echo "=========================================="
echo ""
echo "Choose submission mode:"
echo "  1) comprehensive - Single 12-hour sequential job (RECOMMENDED)"
echo "  2) parallel - 4 parallel jobs using job arrays (faster)"
echo "  3) both - Submit both modes"
echo "  4) skip - Setup only, don't submit"
echo ""

read -p "Enter choice [1-4] (default: 1): " CHOICE
CHOICE=${CHOICE:-1}

case $CHOICE in
    1)
        echo ""
        echo "Submitting comprehensive ablation study..."
        JOB_ID=$(sbatch slurm_ablation_comprehensive.sh | awk '{print $4}')
        echo ""
        echo "=========================================="
        echo "✓ Job Submitted Successfully!"
        echo "=========================================="
        echo "Job ID: $JOB_ID"
        echo ""
        echo "Monitor with:"
        echo "  squeue -j $JOB_ID"
        echo "  tail -f logs/ablation_study_${JOB_ID}.out"
        echo ""
        echo "Cancel with:"
        echo "  scancel $JOB_ID"
        ;;

    2)
        echo ""
        echo "Submitting parallel ablation studies (4 jobs)..."
        JOB_ID=$(sbatch slurm_ablation_parallel.sh | awk '{print $4}')
        echo ""
        echo "=========================================="
        echo "✓ Job Array Submitted Successfully!"
        echo "=========================================="
        echo "Job Array ID: $JOB_ID"
        echo "  Task 0: Embedding ablation"
        echo "  Task 1: Model ablation"
        echo "  Task 2: Ensemble ablation"
        echo "  Task 3: Training ablation"
        echo ""
        echo "Monitor with:"
        echo "  squeue -j $JOB_ID"
        echo "  tail -f logs/ablation_parallel_${JOB_ID}_*.out"
        echo ""
        echo "Cancel with:"
        echo "  scancel $JOB_ID"
        ;;

    3)
        echo ""
        echo "Submitting both comprehensive and parallel jobs..."
        COMP_JOB=$(sbatch slurm_ablation_comprehensive.sh | awk '{print $4}')
        PAR_JOB=$(sbatch slurm_ablation_parallel.sh | awk '{print $4}')
        echo ""
        echo "=========================================="
        echo "✓ Both Jobs Submitted Successfully!"
        echo "=========================================="
        echo "Comprehensive Job ID: $COMP_JOB"
        echo "Parallel Job ID: $PAR_JOB"
        echo ""
        echo "Monitor with:"
        echo "  squeue -u $USER"
        ;;

    4)
        echo ""
        echo "Skipping job submission."
        echo ""
        echo "To submit later, run:"
        echo "  cd $PROJECT_DIR"
        echo "  ./submit_ablation_studies.sh [comprehensive|parallel|both]"
        echo ""
        echo "Or use SLURM directly:"
        echo "  sbatch slurm_ablation_comprehensive.sh"
        echo "  sbatch slurm_ablation_parallel.sh"
        ;;

    *)
        echo "Invalid choice. No jobs submitted."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Useful commands:"
echo "  squeue -u $USER          # Check your job queue"
echo "  scancel -u $USER         # Cancel all your jobs"
echo "  ls -lh results/ablation/ # View results"
echo "  tail -f logs/ablation_*  # Follow logs"
echo ""
