#!/bin/bash

# ==============================================================================
# Deploy AMP Ablation Studies to HPC and Submit Jobs
# ==============================================================================
# This script:
# 1. Syncs the project to the HPC cluster
# 2. Sets up the environment
# 3. Submits ablation study jobs
# ==============================================================================

# HPC Configuration
HPC_USER="pawan"
HPC_HOST="10.240.60.36"
HPC_PROJECT_DIR="/export/home/pawan/amp_prediction"
LOCAL_PROJECT_DIR="D:/Github/Ensemble-Protein-Embedding-Framework-for-AMP-Prediction-2"

echo "=========================================="
echo "AMP ABLATION STUDY DEPLOYMENT"
echo "=========================================="
echo "Local: $LOCAL_PROJECT_DIR"
echo "Remote: $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR"
echo "Time: $(date)"
echo ""

# Check SSH connectivity
echo "Testing SSH connection..."
ssh -o ConnectTimeout=10 $HPC_USER@$HPC_HOST "echo 'SSH connection successful'" || {
    echo "Error: Cannot connect to $HPC_HOST"
    exit 1
}
echo ""

# Create remote directory structure
echo "Creating remote directory structure..."
ssh $HPC_USER@$HPC_HOST "mkdir -p $HPC_PROJECT_DIR/logs $HPC_PROJECT_DIR/results/ablation"
echo ""

# Sync project files to HPC
echo "Syncing project files to HPC..."
rsync -avz --progress \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.git/' \
    --exclude 'legacy/' \
    --exclude 'data/embeddings/*.pt' \
    --exclude 'models/*.pt' \
    "$LOCAL_PROJECT_DIR/" "$HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/"

SYNC_STATUS=$?
if [ $SYNC_STATUS -ne 0 ]; then
    echo "Error: Failed to sync files (exit code: $SYNC_STATUS)"
    exit 1
fi
echo ""

# Set permissions
echo "Setting file permissions..."
ssh $HPC_USER@$HPC_HOST "chmod +x $HPC_PROJECT_DIR/*.sh"
echo ""

# Setup environment on HPC
echo "Setting up environment on HPC..."
ssh $HPC_USER@$HPC_HOST << 'EOF'
cd ~/amp_prediction

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel

# Install package in development mode
cd amp_prediction
pip install -e ".[dev,viz,tracking]"
cd ..

echo "Environment setup complete!"
EOF

echo ""
echo "=========================================="
echo "DEPLOYMENT COMPLETE"
echo "=========================================="
echo ""

# Ask user which submission mode to use
read -p "Submit ablation studies? [comprehensive/parallel/both/no] (default: comprehensive): " SUBMIT_MODE
SUBMIT_MODE=${SUBMIT_MODE:-comprehensive}

if [ "$SUBMIT_MODE" == "no" ]; then
    echo "Skipping job submission."
    echo ""
    echo "To submit manually, run on HPC:"
    echo "  ssh $HPC_USER@$HPC_HOST"
    echo "  cd $HPC_PROJECT_DIR"
    echo "  ./submit_ablation_studies.sh [comprehensive|parallel|both]"
    exit 0
fi

# Submit jobs
echo ""
echo "Submitting ablation studies in mode: $SUBMIT_MODE"
ssh $HPC_USER@$HPC_HOST "cd $HPC_PROJECT_DIR && ./submit_ablation_studies.sh $SUBMIT_MODE"

echo ""
echo "=========================================="
echo "JOBS SUBMITTED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Monitor jobs:"
echo "  ssh $HPC_USER@$HPC_HOST 'squeue -u $HPC_USER'"
echo ""
echo "View logs:"
echo "  ssh $HPC_USER@$HPC_HOST 'tail -f $HPC_PROJECT_DIR/logs/ablation_*.out'"
echo ""
echo "Download results when complete:"
echo "  rsync -avz $HPC_USER@$HPC_HOST:$HPC_PROJECT_DIR/results/ablation/ ./results/ablation/"
echo ""
