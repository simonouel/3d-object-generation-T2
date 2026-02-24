# ============================================================
# 3D Object Generation - Full Environment Setup Script
# ============================================================
# This script installs all prerequisites and sets up the project
# Run as Administrator for best results
# ============================================================

param(
    [string]$InstallPath = "C:\3d-object-generation",
    [string]$CondaEnvName = "3dwithtrellis",
    [string]$GitLabRepo = "https://github.com/NVIDIA-AI-Blueprints/3d-object-generation.git",
    [string]$Branch = "main"
)

# Colors for output
function Write-Step { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Warning { param($msg) Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Write-Error { param($msg) Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Check if running as Administrator
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Warning "Not running as Administrator. Some installations may fail."
    Write-Host "Consider re-running this script as Administrator for full functionality."
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne 'y') { exit 1 }
}

# Check if running from inside the install path (will cause issues with delete)
$currentPath = (Get-Location).Path
if ($currentPath -like "$InstallPath*") {
    Write-Warning "You are running this script from inside the install path!"
    Write-Host "Current location: $currentPath"
    Write-Host "Install path: $InstallPath"
    Write-Host ""
    Write-Host "Please run this script from a different directory, e.g.:"
    Write-Host "  cd C:\"
    Write-Host "  & `"$PSCommandPath`" -InstallPath `"$InstallPath`" -CondaEnvName `"$CondaEnvName`""
    exit 1
}

Write-Host @"

============================================================
  3D Object Generation - Environment Setup
============================================================
  Install Path: $InstallPath
  Conda Env:    $CondaEnvName
  Repository:   $GitLabRepo
  Branch:       $Branch
============================================================

"@ -ForegroundColor White

# ============================================================
# Step 1: Install Git
# ============================================================
Write-Step "Checking Git installation"

$gitInstalled = $null
try {
    $gitInstalled = Get-Command git -ErrorAction SilentlyContinue
} catch {}

if ($gitInstalled) {
    $gitVersion = git --version
    Write-Success "Git is already installed: $gitVersion"
} else {
    Write-Host "Git not found. Installing..."
    
    # Try winget first
    $wingetInstalled = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetInstalled) {
        Write-Host "Installing Git via winget..."
        winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
        
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    } else {
        # Download and install Git manually
        Write-Host "Downloading Git installer..."
        $gitInstallerUrl = "https://github.com/git-for-windows/git/releases/download/v2.43.0.windows.1/Git-2.43.0-64-bit.exe"
        $gitInstallerPath = "$env:TEMP\GitInstaller.exe"
        
        Invoke-WebRequest -Uri $gitInstallerUrl -OutFile $gitInstallerPath -UseBasicParsing
        
        Write-Host "Installing Git (this may take a few minutes)..."
        Start-Process -FilePath $gitInstallerPath -ArgumentList "/VERYSILENT /NORESTART /NOCANCEL /SP- /CLOSEAPPLICATIONS /RESTARTAPPLICATIONS /COMPONENTS=icons,ext\reg\shellhere,assoc,assoc_sh" -Wait
        
        # Refresh PATH
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
        
        Remove-Item $gitInstallerPath -Force -ErrorAction SilentlyContinue
    }
    
    # Verify installation
    try {
        $gitVersion = git --version
        Write-Success "Git installed successfully: $gitVersion"
    } catch {
        Write-Error "Git installation failed. Please install manually from https://git-scm.com/"
        exit 1
    }
}

# ============================================================
# Step 2: Install Git LFS
# ============================================================
Write-Step "Checking Git LFS installation"

$gitLfsInstalled = $null
try {
    $gitLfsInstalled = git lfs version 2>$null
} catch {}

if ($gitLfsInstalled) {
    Write-Success "Git LFS is already installed: $gitLfsInstalled"
} else {
    Write-Host "Git LFS not found. Installing..."
    
    $wingetInstalled = Get-Command winget -ErrorAction SilentlyContinue
    if ($wingetInstalled) {
        Write-Host "Installing Git LFS via winget..."
        winget install --id GitHub.GitLFS -e --source winget --accept-package-agreements --accept-source-agreements
    } else {
        # Download and install Git LFS manually
        Write-Host "Downloading Git LFS installer..."
        $lfsInstallerUrl = "https://github.com/git-lfs/git-lfs/releases/download/v3.4.1/git-lfs-windows-v3.4.1.exe"
        $lfsInstallerPath = "$env:TEMP\GitLfsInstaller.exe"
        
        Invoke-WebRequest -Uri $lfsInstallerUrl -OutFile $lfsInstallerPath -UseBasicParsing
        
        Write-Host "Installing Git LFS..."
        Start-Process -FilePath $lfsInstallerPath -ArgumentList "/VERYSILENT /NORESTART" -Wait
        
        Remove-Item $lfsInstallerPath -Force -ErrorAction SilentlyContinue
    }
    
    # Initialize Git LFS
    git lfs install
    
    # Verify installation
    try {
        $lfsVersion = git lfs version
        Write-Success "Git LFS installed successfully: $lfsVersion"
    } catch {
        Write-Warning "Git LFS installation may have failed. Continuing anyway..."
    }
}

# ============================================================
# Step 3: Install Miniconda
# ============================================================
Write-Step "Checking Conda installation"

$condaInstalled = $null
$condaPath = ""

# Check common Conda locations
$condaPaths = @(
    "$env:USERPROFILE\miniconda3",
    "$env:USERPROFILE\anaconda3",
    "C:\miniconda3",
    "C:\anaconda3",
    "$env:LOCALAPPDATA\miniconda3",
    "$env:ProgramData\miniconda3"
)

foreach ($path in $condaPaths) {
    if (Test-Path "$path\Scripts\conda.exe") {
        $condaPath = $path
        break
    }
}

if ($condaPath) {
    Write-Success "Conda found at: $condaPath"
    
    # Add to PATH for this session
    $env:Path = "$condaPath;$condaPath\Scripts;$condaPath\Library\bin;" + $env:Path
} else {
    Write-Host "Conda not found. Installing Miniconda..."
    
    $minicondaUrl = "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    $minicondaInstaller = "$env:TEMP\Miniconda3Installer.exe"
    $condaPath = "$env:USERPROFILE\miniconda3"
    
    Write-Host "Downloading Miniconda (this may take a few minutes)..."
    Invoke-WebRequest -Uri $minicondaUrl -OutFile $minicondaInstaller -UseBasicParsing
    
    Write-Host "Installing Miniconda to $condaPath..."
    Start-Process -FilePath $minicondaInstaller -ArgumentList "/InstallationType=JustMe /RegisterPython=0 /S /D=$condaPath" -Wait
    
    Remove-Item $minicondaInstaller -Force -ErrorAction SilentlyContinue
    
    # Add to PATH
    $env:Path = "$condaPath;$condaPath\Scripts;$condaPath\Library\bin;" + $env:Path
    
    # Verify installation
    if (Test-Path "$condaPath\Scripts\conda.exe") {
        Write-Success "Miniconda installed successfully at: $condaPath"
    } else {
        Write-Error "Miniconda installation failed. Please install manually from https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    }
}

# Initialize conda for PowerShell
Write-Host "Initializing Conda for PowerShell..."
& "$condaPath\Scripts\conda.exe" init powershell 2>$null

# ============================================================
# Step 4: Clone Repository
# ============================================================
Write-Step "Cloning repository"

# Create parent directory if needed
$parentDir = Split-Path $InstallPath -Parent
if (-not (Test-Path $parentDir)) {
    New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
}

if (Test-Path $InstallPath) {
    Write-Warning "Directory already exists: $InstallPath"
    $overwrite = Read-Host "Delete and re-clone? (y/n)"
    if ($overwrite -eq 'y') {
        Write-Host "Removing existing directory..."
        # Change to parent directory first to avoid "in use" error
        $parentDir = Split-Path $InstallPath -Parent
        Set-Location $parentDir
        
        # Try to remove, handle if still in use
        try {
            Remove-Item -Recurse -Force $InstallPath -ErrorAction Stop
            Write-Success "Directory removed"
        } catch {
            Write-Error "Cannot remove directory - it may be in use."
            Write-Host "Please close any applications using files in: $InstallPath"
            Write-Host "Then run this script again from a different directory."
            exit 1
        }
    } else {
        Write-Host "Using existing directory..."
    }
}

if (-not (Test-Path $InstallPath)) {
    Write-Host "Cloning repository..."
    Write-Host "Repository: $GitLabRepo"
    Write-Host "Branch: $Branch"
    
    git clone -b $Branch $GitLabRepo $InstallPath
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Failed to clone repository. Check your network connection and credentials."
        Write-Host ""
        exit 1
    }
    
    Write-Success "Repository cloned successfully"
} else {
    Write-Success "Using existing repository at: $InstallPath"
}

# Verify the install path exists and has required files
if (-not (Test-Path $InstallPath)) {
    Write-Error "Install path does not exist: $InstallPath"
    exit 1
}

if (-not (Test-Path (Join-Path $InstallPath "install.bat"))) {
    Write-Error "install.bat not found in $InstallPath - this doesn't appear to be a valid repository"
    exit 1
}

# ============================================================
# Step 5: Update config.py with Conda environment name
# ============================================================
Write-Step "Updating config.py with Conda environment name"

Set-Location $InstallPath

$configPath = Join-Path $InstallPath "config.py"
if (Test-Path $configPath) {
    $configContent = Get-Content $configPath -Raw
    
    # Replace CONDA_ENV_NAME value
    $updatedConfig = $configContent -replace 'CONDA_ENV_NAME\s*=\s*[''"][^''\"]*[''"]', "CONDA_ENV_NAME = `"$CondaEnvName`""
    
    Set-Content -Path $configPath -Value $updatedConfig -NoNewline
    Write-Success "Updated CONDA_ENV_NAME to: $CondaEnvName"
} else {
    Write-Warning "config.py not found. Skipping environment name update."
}

# ============================================================
# Step 6: Run install.bat
# ============================================================
Write-Step "Running install.bat"

if (Test-Path "install.bat") {
    Write-Host "Starting installation (this may take 15-30 minutes)..."
    Write-Host "The script will create a Conda environment and install all dependencies."
    Write-Host ""
    
    # Run install.bat
    cmd /c "install.bat"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Installation completed successfully!"
    } else {
        Write-Warning "install.bat finished with exit code: $LASTEXITCODE"
        Write-Host "Check the output above for any errors."
    }
} else {
    Write-Error "install.bat not found in $InstallPath"
    exit 1
}

# ============================================================
# Final Instructions
# ============================================================
Write-Host @"

============================================================
  Setup Complete!
============================================================

To start using the application:

1. Open a new PowerShell or Command Prompt
2. Navigate to: $InstallPath
3. Activate the environment:
   conda activate $CondaEnvName
4. Run the application:
   python app.py

The application will be available at: http://localhost:7860

============================================================
"@ -ForegroundColor Green

Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

