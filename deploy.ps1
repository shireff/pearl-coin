$HOST_IP = "27.77.59.93"
$SSH_PORT = "32519"
$SSH_USER = "root"
$REMOTE_DIR = "/root/pearl"
$LOCAL_DIR = "D:/Front-End Projects/mining/pearl"
$SSH_KEY = "$env:USERPROFILE/.ssh/id_ed25519"

function Deploy {
    Write-Host "=== Pearl Miner Auto-Deploy ===" -ForegroundColor Cyan

    # Check if SSH key exists
    if (-not (Test-Path $SSH_KEY)) {
        Write-Host "ERROR: SSH key not found at $SSH_KEY" -ForegroundColor Red
        Write-Host "Generate it with: ssh-keygen -t ed25519 -C 'pearl-miner'" -ForegroundColor Yellow
        return
    }

    # Step 1: Copy changed files to instance
    Write-Host "`n[1/4] Copying files to instance..." -ForegroundColor Yellow
    scp -P $SSH_PORT -i $SSH_KEY -r "$LOCAL_DIR/miner" "$SSH_USER@$HOST_IP:$REMOTE_DIR/"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to copy files. Check SSH key and instance status." -ForegroundColor Red
        Write-Host "Try: ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$HOST_IP" -ForegroundColor Yellow
        return
    }
    Write-Host "Files copied successfully." -ForegroundColor Green

    # Step 2: Install dependencies on instance
    Write-Host "`n[2/4] Installing dependencies on instance..." -ForegroundColor Yellow
    ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$HOST_IP "cd $REMOTE_DIR && uv sync --all-packages 2>&1 | tail -5"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: uv sync may have failed. Check the output above." -ForegroundColor Yellow
    }
    Write-Host "Dependencies installed." -ForegroundColor Green

    # Step 3: Verify CUDA extension
    Write-Host "`n[3/4] Verifying CUDA extension..." -ForegroundColor Yellow
    ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$HOST_IP "cd $REMOTE_DIR && python -c `"import pearl_gemm_cuda; print('CUDA extension OK')`""
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: CUDA extension check failed. The extension may need to be rebuilt." -ForegroundColor Yellow
    }
    Write-Host "CUDA extension verified." -ForegroundColor Green

    # Step 4: Start mining
    Write-Host "`n[4/4] Starting mining..." -ForegroundColor Yellow
    ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$HOST_IP "cd $REMOTE_DIR/miner && python run_mining.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: Mining may have exited. Check the output above." -ForegroundColor Yellow
    }

    Write-Host "`n=== Deploy Complete ===" -ForegroundColor Green
}

function DeployQuick {
    Write-Host "=== Quick Deploy (git pull + run) ===" -ForegroundColor Cyan
    ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$HOST_IP "cd $REMOTE_DIR && git pull && uv sync --all-packages 2>&1 | tail -3 && cd miner && python run_mining.py"
}

function ShowHelp {
    Write-Host ""
    Write-Host "Usage: .\deploy.ps1 [command]" -ForegroundColor White
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor White
    Write-Host "  (no args)     Full deploy: copy files, sync, verify, run" -ForegroundColor Gray
    Write-Host "  quick         Quick deploy: git pull + sync + run" -ForegroundColor Gray
    Write-Host "  check         Check SSH connection and CUDA extension" -ForegroundColor Gray
    Write-Host "  status        Check instance status" -ForegroundColor Gray
    Write-Host ""
}

# Parse arguments
switch ($args[0]) {
    "quick" { DeployQuick }
    "check" {
        Write-Host "Checking SSH connection..." -ForegroundColor Yellow
        ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$HOST_IP "echo 'SSH OK' && python -c `"import pearl_gemm_cuda; print('CUDA extension OK')`""
    }
    "status" {
        Write-Host "Instance: $HOST_IP:$SSH_PORT" -ForegroundColor White
        ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$HOST_IP "echo 'Instance is running' && nvidia-smi 2>/dev/null | head -5"
    }
    default { Deploy }
}