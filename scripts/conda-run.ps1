# Helper script to run commands in conda environment
# Used by VS Code tasks when conda is not directly available in PowerShell

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Command,
    
    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"

# Find conda.exe
$condaExe = $null
$condaPaths = @(
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\Anaconda3\Scripts\conda.exe",
    "C:\ProgramData\Miniconda3\Scripts\conda.exe",
    "C:\Anaconda3\Scripts\conda.exe",
    "C:\Miniconda3\Scripts\conda.exe"
)

foreach ($path in $condaPaths) {
    if (Test-Path $path) {
        $condaExe = $path
        break
    }
}

# Also check if conda is in PATH
if (-not $condaExe) {
    $condaCmd = Get-Command conda.exe -ErrorAction SilentlyContinue
    if ($condaCmd) {
        $condaExe = $condaCmd.Source
    }
}

if (-not $condaExe) {
    Write-Host "Error: Could not find conda.exe" -ForegroundColor Red
    Write-Host "Please ensure Anaconda or Miniconda is installed" -ForegroundColor Yellow
    exit 1
}

# Get conda environment path
$condaEnvPath = $null
try {
    $envListJson = & $condaExe env list --json 2>&1
    if ($LASTEXITCODE -eq 0) {
        $envList = $envListJson | ConvertFrom-Json
        $condaEnvPath = $envList.envs | Where-Object { $_ -like "*rayspace3d*" } | Select-Object -First 1
    }
} catch {
    # Fallback: try common locations
    $condaRoot = Split-Path -Parent (Split-Path -Parent $condaExe)
    $commonPaths = @(
        (Join-Path $condaRoot "envs\rayspace3d"),
        (Join-Path $env:USERPROFILE ".conda\envs\rayspace3d")
    )
    foreach ($path in $commonPaths) {
        if (Test-Path $path) {
            $condaEnvPath = $path
            break
        }
    }
}

if (-not $condaEnvPath -or -not (Test-Path $condaEnvPath)) {
    Write-Host "Error: rayspace3d conda environment not found" -ForegroundColor Red
    Write-Host "Please create it first: conda env create -f environment.yml" -ForegroundColor Yellow
    exit 1
}

# Set up environment variables
$env:CONDA_PREFIX = $condaEnvPath
$env:CONDA_DEFAULT_ENV = "rayspace3d"
$condaBinPath = Join-Path $condaEnvPath "Scripts"
$condaLibPath = Join-Path $condaEnvPath "Library\bin"
if (Test-Path $condaBinPath) {
    $env:PATH = "$condaBinPath;$condaLibPath;$env:PATH"
}
$env:CMAKE_PREFIX_PATH = $condaEnvPath
$env:CUDA_PATH = $condaEnvPath

# Run the command with all arguments
& $Command $Arguments
exit $LASTEXITCODE

