#!/bin/bash

# Simple deployment using scp (no rsync required)
# This works on Windows Git Bash and WSL

HPC_USER="pawan"
HPC_HOST="10.240.60.36"
HPC_DIR="/export/home/pawan/amp_prediction"

echo "=========================================="
echo "Deploying AMP Ablation Studies to HPC"
echo "=========================================="

# Test connection
echo "Testing SSH connection..."
ssh -o ConnectTimeout=10 ${HPC_USER}@${HPC_HOST} "echo 'Connected'" || exit 1
echo ""

# Create directories
echo "Creating remote directories..."
ssh ${HPC_USER}@${HPC_HOST} "mkdir -p ${HPC_DIR}/logs ${HPC_DIR}/results/ablation"

# Copy SLURM scripts
echo "Copying SLURM scripts..."
scp slurm_ablation_comprehensive.sh ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/
scp slurm_ablation_parallel.sh ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/
scp submit_ablation_studies.sh ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/

# Copy amp_prediction directory
echo "Copying project files..."
scp -r amp_prediction ${HPC_USER}@${HPC_HOST}:${HPC_DIR}/

# Set permissions
echo "Setting permissions..."
ssh ${HPC_USER}@${HPC_HOST} "chmod +x ${HPC_DIR}/*.sh"

# Setup environment
echo "Setting up Python environment..."
ssh ${HPC_USER}@${HPC_HOST} << 'ENDSSH'
cd ~/amp_prediction
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip
cd amp_prediction
pip install -e ".[dev,viz,tracking]"
cd ..
echo "Environment ready!"
ENDSSH

echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Submit jobs with:"
echo "  ssh ${HPC_USER}@${HPC_HOST}"
echo "  cd ${HPC_DIR}"
echo "  ./submit_ablation_studies.sh comprehensive"
echo ""
