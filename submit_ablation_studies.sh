#!/bin/bash

# ==============================================================================
# Submit Ablation Studies to SLURM Cluster
# ==============================================================================
# This script submits ablation study jobs to the SLURM cluster
# Usage: ./submit_ablation_studies.sh [comprehensive|parallel|both]
# ==============================================================================

MODE=${1:-comprehensive}

echo "=========================================="
echo "AMP ABLATION STUDY SUBMISSION"
echo "=========================================="
echo "Mode: $MODE"
echo "Time: $(date)"
echo ""

# Create logs directory
mkdir -p logs

case $MODE in
    comprehensive)
        echo "Submitting comprehensive ablation study..."
        echo "This runs all ablation studies sequentially in a single job"
        JOB_ID=$(sbatch slurm_ablation_comprehensive.sh | awk '{print $4}')
        echo "Job submitted: $JOB_ID"
        echo ""
        echo "Monitor with:"
        echo "  squeue -u $USER -j $JOB_ID"
        echo "  tail -f logs/ablation_study_${JOB_ID}.out"
        ;;

    parallel)
        echo "Submitting parallel ablation studies..."
        echo "This runs 4 ablation studies in parallel using job arrays"
        JOB_ID=$(sbatch slurm_ablation_parallel.sh | awk '{print $4}')
        echo "Job array submitted: $JOB_ID"
        echo "  - Task 0: Embedding ablation"
        echo "  - Task 1: Model ablation"
        echo "  - Task 2: Ensemble ablation"
        echo "  - Task 3: Training ablation"
        echo ""
        echo "Monitor with:"
        echo "  squeue -u $USER -j $JOB_ID"
        echo "  tail -f logs/ablation_parallel_${JOB_ID}_*.out"
        ;;

    both)
        echo "Submitting both comprehensive and parallel jobs..."
        echo ""
        echo "1. Comprehensive (sequential):"
        COMP_JOB_ID=$(sbatch slurm_ablation_comprehensive.sh | awk '{print $4}')
        echo "   Job ID: $COMP_JOB_ID"
        echo ""
        echo "2. Parallel (job array):"
        PAR_JOB_ID=$(sbatch slurm_ablation_parallel.sh | awk '{print $4}')
        echo "   Job ID: $PAR_JOB_ID"
        echo ""
        echo "Monitor all jobs:"
        echo "  squeue -u $USER"
        ;;

    *)
        echo "Error: Invalid mode '$MODE'"
        echo "Usage: $0 [comprehensive|parallel|both]"
        echo ""
        echo "Options:"
        echo "  comprehensive - Run all studies sequentially (12h single job)"
        echo "  parallel      - Run studies in parallel (4 jobs, faster overall)"
        echo "  both          - Submit both modes for comparison"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Submission complete!"
echo "=========================================="
echo ""
echo "Commands:"
echo "  Check queue:      squeue -u $USER"
echo "  Cancel jobs:      scancel -u $USER"
echo "  View output:      tail -f logs/ablation_*.out"
echo "  Check results:    ls -lh results/ablation/"
echo ""
