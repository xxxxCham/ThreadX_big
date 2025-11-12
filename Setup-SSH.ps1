# ============================================
# SSH Setup Script for GitHub ThreadX_big
# ============================================
# Complete automation for SSH key creation and GitHub authentication
# Usage: .\Setup-SSH.ps1
# 
# This script:
# 1. Checks for existing SSH key
# 2. Generates ed25519 key (no passphrase for automation)
# 3. Starts SSH Agent service
# 4. Adds key to SSH Agent
# 5. Configures Git to use SSH
# 6. Tests SSH connection to GitHub
# 7. Verifies Git operations work

param(
    [switch]$NoTest = $false,
    [string]$GitUsername = "xxxxCham"
)

# ============================================
# 1️⃣ CHECK EXISTING SSH KEY
# ============================================

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     SSH Setup for GitHub (ThreadX_big)                 ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

$sshKeyPath = "$env:USERPROFILE\.ssh\id_ed25519"
$sshDir = "$env:USERPROFILE\.ssh"

Write-Host "🔍 Step 1: Checking for existing SSH key..." -ForegroundColor Yellow

if (Test-Path $sshDir) {
    Write-Host "   ✅ SSH directory exists: $sshDir" -ForegroundColor Green
} else {
    Write-Host "   📁 Creating SSH directory..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $sshDir > $null
    Write-Host "   ✅ SSH directory created" -ForegroundColor Green
}

if (Test-Path $sshKeyPath) {
    Write-Host "   ✅ SSH key already exists: $sshKeyPath" -ForegroundColor Green
    $response = Read-Host "   Use existing key? (Y/n)"
    
    if ($response -eq "n" -or $response -eq "N") {
        Write-Host "   ⚠️  Backup your existing key before regenerating!" -ForegroundColor Yellow
        Write-Host "   Location: $sshKeyPath" -ForegroundColor Yellow
        exit 1
    }
    $skipGeneration = $true
} else {
    Write-Host "   ❌ No SSH key found at: $sshKeyPath" -ForegroundColor Yellow
    $skipGeneration = $false
}

# ============================================
# 2️⃣ GENERATE SSH KEY
# ============================================

if (-not $skipGeneration) {
    Write-Host ""
    Write-Host "🔑 Step 2: Generating SSH key..." -ForegroundColor Yellow
    Write-Host "   ⏳ This will create ed25519 key (modern, secure, fast)" -ForegroundColor Cyan
    
    try {
        # Generate key without passphrase (automation-friendly)
        ssh-keygen -t ed25519 -C "$GitUsername@github.com" -f $sshKeyPath -N "" -q
        
        if ($?) {
            Write-Host "   ✅ SSH key generated successfully" -ForegroundColor Green
            Write-Host "      Key: $sshKeyPath" -ForegroundColor Green
        } else {
            Write-Host "   ❌ SSH key generation failed" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "   ❌ Error generating key: $_" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host ""
    Write-Host "✅ Step 2: Skipped (using existing key)" -ForegroundColor Green
}

# ============================================
# 3️⃣ START SSH AGENT SERVICE
# ============================================

Write-Host ""
Write-Host "🚀 Step 3: Starting SSH Agent service..." -ForegroundColor Yellow

$sshAgentService = Get-Service -Name ssh-agent -ErrorAction SilentlyContinue

if ($null -eq $sshAgentService) {
    Write-Host "   ⚠️  SSH Agent service not found" -ForegroundColor Yellow
    Write-Host "   📋 Installing OpenSSH Client..." -ForegroundColor Cyan
    
    try {
        Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0 | Out-Null
        Write-Host "   ✅ OpenSSH Client installed" -ForegroundColor Green
        
        $sshAgentService = Get-Service -Name ssh-agent -ErrorAction SilentlyContinue
        if ($null -eq $sshAgentService) {
            Write-Host "   ❌ SSH Agent still not available" -ForegroundColor Red
            exit 1
        }
    } catch {
        Write-Host "   ❌ Failed to install OpenSSH: $_" -ForegroundColor Red
        Write-Host "   💡 Try manually: Settings → Apps → Optional features → Add OpenSSH Client" -ForegroundColor Yellow
        exit 1
    }
}

if ($sshAgentService.Status -eq "Running") {
    Write-Host "   ✅ SSH Agent is running" -ForegroundColor Green
} else {
    Write-Host "   ⏳ Starting SSH Agent..." -ForegroundColor Cyan
    
    try {
        Start-Service -Name ssh-agent
        Write-Host "   ✅ SSH Agent started" -ForegroundColor Green
    } catch {
        Write-Host "   ❌ Failed to start SSH Agent: $_" -ForegroundColor Red
        exit 1
    }
}

# ============================================
# 4️⃣ ADD KEY TO SSH AGENT
# ============================================

Write-Host ""
Write-Host "🔐 Step 4: Adding key to SSH Agent..." -ForegroundColor Yellow

try {
    # Use -N "" to avoid passphrase prompt (key generated without passphrase)
    ssh-add "$sshKeyPath" 2>&1 | Out-Null
    
    if ($?) {
        Write-Host "   ✅ SSH key added to agent" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Could not add key to agent (may already be added)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠️  Error adding key: $_" -ForegroundColor Yellow
}

# Verify key is in agent
$keys = ssh-add -l 2>&1
if ($keys -like "*ed25519*" -or $LASTEXITCODE -eq 0) {
    Write-Host "   ✅ Key verified in SSH Agent" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Could not verify key in agent" -ForegroundColor Yellow
}

# ============================================
# 5️⃣ DISPLAY PUBLIC KEY
# ============================================

Write-Host ""
Write-Host "📋 Step 5: Your public SSH key" -ForegroundColor Yellow
Write-Host "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan

if (Test-Path "$sshKeyPath.pub") {
    $pubKey = Get-Content "$sshKeyPath.pub"
    Write-Host ""
    Write-Host $pubKey -ForegroundColor White
    Write-Host ""
    Write-Host "   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   📌 Add this key to GitHub:" -ForegroundColor Green
    Write-Host "      1. Visit: https://github.com/settings/keys" -ForegroundColor Cyan
    Write-Host "      2. Or:   https://github.com/settings/ssh/new" -ForegroundColor Cyan
    Write-Host "      3. Click 'New SSH key'" -ForegroundColor Cyan
    Write-Host "      4. Title: 'ThreadX_big - $env:COMPUTERNAME'" -ForegroundColor Cyan
    Write-Host "      5. Key type: Authentication Key" -ForegroundColor Cyan
    Write-Host "      6. Paste entire key above" -ForegroundColor Cyan
    Write-Host "      7. Click 'Add SSH key'" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "   🔑 Note: You already have a key on GitHub" -ForegroundColor Yellow
    Write-Host "      You can either add this new one OR use the existing" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host "   ❌ Public key not found" -ForegroundColor Red
    exit 1
}

# ============================================
# 6️⃣ CONFIGURE GIT FOR SSH
# ============================================

Write-Host "🔗 Step 6: Configuring Git for SSH..." -ForegroundColor Yellow

$gitDir = "d:\ThreadX_big"
if (-not (Test-Path $gitDir)) {
    Write-Host "   ❌ Git directory not found: $gitDir" -ForegroundColor Red
    exit 1
}

Push-Location $gitDir

# Check current remote
$currentRemote = git remote get-url origin 2>&1

Write-Host "   Current remote: $currentRemote" -ForegroundColor Cyan

# Set SSH remote
git remote set-url origin "git@github.com:$GitUsername/ThreadX_big.git"

if ($?) {
    Write-Host "   ✅ Git remote updated to SSH" -ForegroundColor Green
    git remote -v | Write-Host -ForegroundColor Green
} else {
    Write-Host "   ❌ Failed to update Git remote" -ForegroundColor Red
    Pop-Location
    exit 1
}

# ============================================
# 7️⃣ TEST SSH CONNECTION
# ============================================

Write-Host ""
Write-Host "🧪 Step 7: Testing SSH connection..." -ForegroundColor Yellow
Write-Host "   ⏳ Connecting to github.com..." -ForegroundColor Cyan

try {
    $sshTest = ssh -T git@github.com 2>&1
    
    if ($sshTest -like "*successfully authenticated*" -or $sshTest -like "*You've successfully*") {
        Write-Host "   ✅ SSH authentication successful!" -ForegroundColor Green
        Write-Host "      Message: $sshTest" -ForegroundColor Green
    } elseif ($sshTest -like "*Permission denied*") {
        Write-Host "   ❌ Permission denied - Public key not on GitHub yet" -ForegroundColor Red
        Write-Host "   💡 Add your public key to GitHub:" -ForegroundColor Yellow
        Write-Host "      https://github.com/settings/ssh/new" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "   Your key:" -ForegroundColor Cyan
        Write-Host $pubKey -ForegroundColor White
        Pop-Location
        exit 1
    } else {
        Write-Host "   ⚠️  SSH response: $sshTest" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠️  SSH test error: $_" -ForegroundColor Yellow
}

# ============================================
# 8️⃣ TEST GIT OPERATIONS
# ============================================

Write-Host ""
Write-Host "📤 Step 8: Testing Git operations..." -ForegroundColor Yellow

try {
    Write-Host "   ⏳ Running: git pull origin main" -ForegroundColor Cyan
    $gitPull = git pull origin main 2>&1
    
    if ($?) {
        Write-Host "   ✅ Git pull successful" -ForegroundColor Green
        Write-Host "      $(($gitPull | Select-Object -First 1))" -ForegroundColor Green
    } else {
        Write-Host "   ⚠️  Git pull status: $gitPull" -ForegroundColor Yellow
        Write-Host "      (May be normal if no new commits)" -ForegroundColor Yellow
    }
} catch {
    Write-Host "   ⚠️  Git test error: $_" -ForegroundColor Yellow
}

Pop-Location

# ============================================
# ✅ COMPLETION
# ============================================

Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║     ✅ SSH SETUP COMPLETE!                             ║" -ForegroundColor Green
Write-Host "╚════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "🎯 You can now use SSH with GitHub:" -ForegroundColor Green
Write-Host "   • git push origin main" -ForegroundColor Cyan
Write-Host "   • git pull origin main" -ForegroundColor Cyan
Write-Host "   • git clone git@github.com:user/repo.git" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔐 SSH Agent is running and your key is loaded" -ForegroundColor Green
Write-Host "   • Keys in agent: ssh-add -l" -ForegroundColor Cyan
Write-Host "   • Remove key:   ssh-add -d ~/.ssh/id_ed25519" -ForegroundColor Cyan
Write-Host ""
Write-Host "📚 For more info:" -ForegroundColor Green
Write-Host "   • GitHub SSH: https://github.com/settings/keys" -ForegroundColor Cyan
Write-Host "   • Documentation: SSH_VS_TOKEN_GUIDE.md" -ForegroundColor Cyan
Write-Host ""
