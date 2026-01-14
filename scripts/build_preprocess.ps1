# Build RaySpace3D Preprocess using conda on Windows

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$SourceDir = Join-Path $ProjectDir "preprocess"
$BuildDir = Join-Path $ProjectDir "build\preprocess"

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

Write-Host "Found conda at: $condaExe" -ForegroundColor Green

# Activate Environment
$EnvName = "rayspace3d_preprocess"
$condaEnvPath = $null

Write-Host "Finding $EnvName conda environment..." -ForegroundColor Cyan

# Get conda root directory and check common paths
$condaRoot = Split-Path -Parent (Split-Path -Parent $condaExe)
$possibleEnvPaths = @(
    (Join-Path $condaRoot "envs\$EnvName"),
    (Join-Path $env:USERPROFILE ".conda\envs\$EnvName"),
    (Join-Path $env:USERPROFILE "anaconda3\envs\$EnvName"),
    (Join-Path $env:USERPROFILE "miniconda3\envs\$EnvName")
)

foreach ($path in $possibleEnvPaths) {
    if (Test-Path $path) {
        $condaEnvPath = $path
        break
    }
}

if (-not $condaEnvPath) {
    # Try via conda env list as fallback
    try {
        $envListJson = & $condaExe env list --json 2>&1
        if ($LASTEXITCODE -eq 0) {
            $envList = $envListJson | ConvertFrom-Json
            $condaEnvPath = $envList.envs | Where-Object { $_ -like "*$EnvName*" } | Select-Object -First 1
        }
    } catch {}
}

if (-not $condaEnvPath -or -not (Test-Path $condaEnvPath)) {
    Write-Host "Error: $EnvName conda environment not found" -ForegroundColor Red
    Write-Host "Please create it first:" -ForegroundColor Yellow
    Write-Host "  cd preprocess" -ForegroundColor Yellow
    Write-Host "  conda env create -f environment.yml" -ForegroundColor Yellow
    exit 1
}

Write-Host "Found conda environment at: $condaEnvPath" -ForegroundColor Green

# Set up environment variables
$env:CONDA_PREFIX = $condaEnvPath
$env:CONDA_DEFAULT_ENV = $EnvName

# Add conda environment to PATH
$condaBinPath = Join-Path $condaEnvPath "Scripts"
$condaLibPath = Join-Path $condaEnvPath "Library\bin"
if (Test-Path $condaBinPath) {
    $env:PATH = "$condaBinPath;$condaLibPath;$env:PATH"
}

$BuildType = if ($env:BUILD_TYPE) { $env:BUILD_TYPE } else { "Debug" }

Write-Host ""
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Building RaySpace3D Preprocess" -ForegroundColor Cyan
Write-Host "Using conda environment: $env:CONDA_PREFIX" -ForegroundColor Cyan
Write-Host "Build type: $BuildType" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Set up library and include paths for conda
$env:CMAKE_PREFIX_PATH = $env:CONDA_PREFIX
$env:CUDA_PATH = $env:CONDA_PREFIX

# Create build directory
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir

Write-Host ""
Write-Host "Running CMake..." -ForegroundColor Cyan
cmake -S $SourceDir -B $BuildDir -DCMAKE_BUILD_TYPE=$BuildType

if ($LASTEXITCODE -ne 0) {
    Write-Host "CMake configuration failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}

Write-Host ""
Write-Host "Building..." -ForegroundColor Cyan
cmake --build $BuildDir --config $BuildType --parallel

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    Pop-Location
    exit 1
}

Pop-Location

Write-Host ""
Write-Host "===================================" -ForegroundColor Green
Write-Host "Build complete!" -ForegroundColor Green
Write-Host "===================================" -ForegroundColor Green
Write-Host "Binaries: $(Join-Path $BuildDir "bin\$BuildType")" -ForegroundColor Green
Write-Host ""
