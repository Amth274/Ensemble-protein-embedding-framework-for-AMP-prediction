#!/bin/bash

HPC="pawan@10.240.60.36"
HPC_DIR="/export/home/pawan/amp_prediction"

echo "Transferring essential files to HPC..."

# Create remote directory
ssh $HPC "mkdir -p $HPC_DIR/logs $HPC_DIR/results/ablation"

# Transfer SLURM scripts
echo "Copying SLURM scripts..."
scp slurm_ablation_comprehensive.sh $HPC:$HPC_DIR/
scp slurm_ablation_parallel.sh $HPC:$HPC_DIR/
scp submit_ablation_studies.sh $HPC:$HPC_DIR/
scp setup_and_run_on_hpc.sh $HPC:$HPC_DIR/

# Transfer amp_prediction directory (essential files only)
echo "Copying amp_prediction directory..."
scp -r amp_prediction/src $HPC:$HPC_DIR/amp_prediction/
scp -r amp_prediction/scripts $HPC:$HPC_DIR/amp_prediction/
scp -r amp_prediction/configs $HPC:$HPC_DIR/amp_prediction/
scp amp_prediction/setup.py $HPC:$HPC_DIR/amp_prediction/
scp amp_prediction/requirements.txt $HPC:$HPC_DIR/amp_prediction/
scp amp_prediction/__init__.py $HPC:$HPC_DIR/amp_prediction/ 2>/dev/null || true

# Make scripts executable
ssh $HPC "chmod +x $HPC_DIR/*.sh"

echo ""
echo "Files transferred! Now running setup on HPC..."
echo ""

# Run setup and submission on HPC
ssh -t $HPC "cd $HPC_DIR && bash setup_and_run_on_hpc.sh"
