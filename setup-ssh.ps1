# setup-ssh.ps1 — First-time SSH setup for Vast.ai instance

$HOST_IP = "27.77.59.93"
$SSH_PORT = "32519"
$SSH_USER = "root"
$SSH_KEY = "$env:USERPROFILE/.ssh/id_ed25519"

Write-Host "=== Vast.ai SSH Setup ===" -ForegroundColor Cyan

# Step 1: Check if SSH key exists
Write-Host "`n[1/3] Checking SSH key..." -ForegroundColor Yellow
if (-not (Test-Path $SSH_KEY)) {
    Write-Host "SSH key not found. Generating one..." -ForegroundColor Yellow
    ssh-keygen -t ed25519 -C "pearl-miner"
} else {
    Write-Host "SSH key found at $SSH_KEY" -ForegroundColor Green
}

# Step 2: Show the public key to add to Vast.ai
Write-Host "`n[2/3] Add this public key to your Vast.ai account:" -ForegroundColor Yellow
Write-Host "Go to: https://cloud.vast.ai/manage-keys/" -ForegroundColor White
Write-Host ""
$pubKey = Get-Content "$SSH_KEY.pub"
Write-Host "Public key:" -ForegroundColor Gray
Write-Host $pubKey -ForegroundColor White
Write-Host ""
Write-Host "Copy the key above and paste it at https://cloud.vast.ai/manage-keys/" -ForegroundColor Yellow
Write-Host "Then press Enter to continue..."
Read-Host ""

# Step 3: Test SSH connection
Write-Host "`n[3/3] Testing SSH connection..." -ForegroundColor Yellow
ssh -p $SSH_PORT -i $SSH_KEY $SSH_USER@$HOST_IP "echo 'SSH connection successful!'"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSSH setup complete! You can now use .\deploy.ps1" -ForegroundColor Green
    Write-Host ""
    Write-Host "Usage:" -ForegroundColor White
    Write-Host "  .\deploy.ps1          - Full deploy" -ForegroundColor Gray
    Write-Host "  .\deploy.ps1 quick    - Quick deploy (git pull + run)" -ForegroundColor Gray
    Write-Host "  .\deploy.ps1 check    - Check SSH + CUDA" -ForegroundColor Gray
    Write-Host "  .\deploy.ps1 status   - Check instance status" -ForegroundColor Gray
} else {
    Write-Host "`nSSH connection failed." -ForegroundColor Red
    Write-Host "Make sure you added the public key to your Vast.ai account." -ForegroundColor Yellow
    Write-Host "If the instance was created before adding the key, you may need to:" -ForegroundColor Yellow
    Write-Host "  1. Delete the instance and create a new one" -ForegroundColor Yellow
    Write-Host "  2. Or use the Vast.ai web SSH button (it handles keys automatically)" -ForegroundColor Yellow
}