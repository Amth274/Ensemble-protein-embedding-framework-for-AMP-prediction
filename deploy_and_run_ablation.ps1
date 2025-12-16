# ==============================================================================
# Deploy AMP Ablation Studies to HPC and Submit Jobs (PowerShell)
# ==============================================================================
# This script:
# 1. Syncs the project to the HPC cluster using rsync/scp
# 2. Sets up the environment
# 3. Submits ablation study jobs
# ==============================================================================

# HPC Configuration
$HPC_USER = "pawan"
$HPC_HOST = "10.240.60.36"
$HPC_PROJECT_DIR = "/export/home/pawan/amp_prediction"
$LOCAL_PROJECT_DIR = "D:\Github\Ensemble-Protein-Embedding-Framework-for-AMP-Prediction-2"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "AMP ABLATION STUDY DEPLOYMENT" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Local: $LOCAL_PROJECT_DIR"
Write-Host "Remote: ${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}"
Write-Host "Time: $(Get-Date)"
Write-Host ""

# Check if ssh is available
if (-not (Get-Command ssh -ErrorAction SilentlyContinue)) {
    Write-Host "Error: SSH not found. Please install OpenSSH or use WSL." -ForegroundColor Red
    exit 1
}

# Test SSH connection
Write-Host "Testing SSH connection..." -ForegroundColor Yellow
try {
    ssh -o ConnectTimeout=10 "${HPC_USER}@${HPC_HOST}" "echo 'SSH connection successful'"
    if ($LASTEXITCODE -ne 0) { throw }
    Write-Host "✓ SSH connection successful" -ForegroundColor Green
} catch {
    Write-Host "✗ Error: Cannot connect to $HPC_HOST" -ForegroundColor Red
    exit 1
}
Write-Host ""

# Create remote directory structure
Write-Host "Creating remote directory structure..." -ForegroundColor Yellow
ssh "${HPC_USER}@${HPC_HOST}" "mkdir -p ${HPC_PROJECT_DIR}/logs ${HPC_PROJECT_DIR}/results/ablation"
Write-Host ""

# Sync project files using rsync (if available) or scp
Write-Host "Syncing project files to HPC..." -ForegroundColor Yellow

if (Get-Command rsync -ErrorAction SilentlyContinue) {
    # Use rsync if available (Git Bash, WSL, or Cygwin)
    Write-Host "Using rsync for file transfer..." -ForegroundColor Cyan

    # Convert Windows path to Unix-style for rsync
    $unixPath = $LOCAL_PROJECT_DIR -replace '\\', '/' -replace 'D:', '/d'

    rsync -avz --progress `
        --exclude 'venv/' `
        --exclude '__pycache__/' `
        --exclude '*.pyc' `
        --exclude '.git/' `
        --exclude 'legacy/' `
        --exclude 'data/embeddings/*.pt' `
        --exclude 'models/*.pt' `
        "${unixPath}/" "${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}/"
} else {
    Write-Host "rsync not found, using scp (this may be slower)..." -ForegroundColor Yellow

    # List of directories to sync
    $dirs = @(
        "amp_prediction",
        "docs",
        "*.sh",
        "*.md",
        "*.py"
    )

    foreach ($dir in $dirs) {
        $sourcePath = Join-Path $LOCAL_PROJECT_DIR $dir
        if (Test-Path $sourcePath) {
            Write-Host "  Copying $dir..." -ForegroundColor Gray
            scp -r $sourcePath "${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}/"
        }
    }
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "✗ Error: Failed to sync files" -ForegroundColor Red
    exit 1
}
Write-Host "✓ Files synced successfully" -ForegroundColor Green
Write-Host ""

# Set permissions
Write-Host "Setting file permissions..." -ForegroundColor Yellow
ssh "${HPC_USER}@${HPC_HOST}" "chmod +x ${HPC_PROJECT_DIR}/*.sh"
Write-Host ""

# Setup environment on HPC
Write-Host "Setting up environment on HPC..." -ForegroundColor Yellow
ssh "${HPC_USER}@${HPC_HOST}" @"
cd ~/amp_prediction

# Create virtual environment if it doesn't exist
if [ ! -d 'venv' ]; then
    echo 'Creating virtual environment...'
    python3 -m venv venv
fi

# Activate and install dependencies
source venv/bin/activate

echo 'Installing dependencies...'
pip install --upgrade pip setuptools wheel

# Install package in development mode
cd amp_prediction
pip install -e '.[dev,viz,tracking]'
cd ..

echo 'Environment setup complete!'
"@

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "DEPLOYMENT COMPLETE" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""

# Ask user which submission mode to use
$submitMode = Read-Host "Submit ablation studies? [comprehensive/parallel/both/no] (default: comprehensive)"
if ([string]::IsNullOrWhiteSpace($submitMode)) {
    $submitMode = "comprehensive"
}

if ($submitMode -eq "no") {
    Write-Host "Skipping job submission." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To submit manually, run on HPC:"
    Write-Host "  ssh ${HPC_USER}@${HPC_HOST}" -ForegroundColor Cyan
    Write-Host "  cd ${HPC_PROJECT_DIR}" -ForegroundColor Cyan
    Write-Host "  ./submit_ablation_studies.sh [comprehensive|parallel|both]" -ForegroundColor Cyan
    exit 0
}

# Submit jobs
Write-Host ""
Write-Host "Submitting ablation studies in mode: $submitMode" -ForegroundColor Yellow
ssh "${HPC_USER}@${HPC_HOST}" "cd ${HPC_PROJECT_DIR} && ./submit_ablation_studies.sh ${submitMode}"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "JOBS SUBMITTED SUCCESSFULLY" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Monitor jobs:" -ForegroundColor Cyan
Write-Host "  ssh ${HPC_USER}@${HPC_HOST} 'squeue -u ${HPC_USER}'"
Write-Host ""
Write-Host "View logs:" -ForegroundColor Cyan
Write-Host "  ssh ${HPC_USER}@${HPC_HOST} 'tail -f ${HPC_PROJECT_DIR}/logs/ablation_*.out'"
Write-Host ""
Write-Host "Download results when complete:" -ForegroundColor Cyan
Write-Host "  scp -r ${HPC_USER}@${HPC_HOST}:${HPC_PROJECT_DIR}/results/ablation/ ./results/ablation/"
Write-Host ""
