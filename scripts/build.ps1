# Build RaySpace3D using conda on Windows
# This script requires conda CUDA toolkit and keeps the conda environment active throughout

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$BuildDir = Join-Path $ProjectDir "build"

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
    Write-Host "Common locations checked:" -ForegroundColor Yellow
    foreach ($path in $condaPaths) {
        Write-Host "  - $path" -ForegroundColor Gray
    }
    exit 1
}

Write-Host "Found conda at: $condaExe" -ForegroundColor Green

# Get conda environment path
$condaEnvPath = $null
if ($env:CONDA_PREFIX -and $env:CONDA_PREFIX -like "*rayspace3d*") {
    $condaEnvPath = $env:CONDA_PREFIX
    Write-Host "Using existing conda environment: $condaEnvPath" -ForegroundColor Green
} else {
    Write-Host "Finding rayspace3d conda environment..." -ForegroundColor Cyan
    
    # Get conda root directory
    $condaRoot = Split-Path -Parent (Split-Path -Parent $condaExe)
    
    # Try to get environment path from conda
    try {
        $envListJson = & $condaExe env list --json 2>&1
        if ($LASTEXITCODE -eq 0) {
            $envList = $envListJson | ConvertFrom-Json
            $condaEnvPath = $envList.envs | Where-Object { $_ -like "*rayspace3d*" } | Select-Object -First 1
        }
    } catch {
        # Fallback: try common locations
        $commonPaths = @(
            (Join-Path $condaRoot "envs\rayspace3d"),
            (Join-Path $env:USERPROFILE ".conda\envs\rayspace3d"),
            (Join-Path $env:USERPROFILE "anaconda3\envs\rayspace3d"),
            (Join-Path $env:USERPROFILE "miniconda3\envs\rayspace3d")
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
        Write-Host "Please create it first:" -ForegroundColor Yellow
        Write-Host "  conda env create -f environment.yml" -ForegroundColor Yellow
        exit 1
    }
    
    Write-Host "Found conda environment at: $condaEnvPath" -ForegroundColor Green
    
    # Set up environment variables that conda would normally set
    $env:CONDA_PREFIX = $condaEnvPath
    $env:CONDA_DEFAULT_ENV = "rayspace3d"
    
    # Add conda environment to PATH
    $condaBinPath = Join-Path $condaEnvPath "Scripts"
    $condaLibPath = Join-Path $condaEnvPath "Library\bin"
    if (Test-Path $condaBinPath) {
        $env:PATH = "$condaBinPath;$condaLibPath;$env:PATH"
    }
}

if (-not $env:CONDA_PREFIX) {
    Write-Host "Error: CONDA_PREFIX not set. Conda environment may not be activated." -ForegroundColor Red
    exit 1
}

$BuildType = if ($env:BUILD_TYPE) { $env:BUILD_TYPE } else { "Release" }

Write-Host ""
Write-Host "===================================" -ForegroundColor Cyan
Write-Host "Building RaySpace3D" -ForegroundColor Cyan
Write-Host "Using conda environment: $env:CONDA_PREFIX" -ForegroundColor Cyan
Write-Host "Build type: $BuildType" -ForegroundColor Cyan
Write-Host "===================================" -ForegroundColor Cyan
Write-Host ""

# Set OptiX path (Windows default location)
$OptiXInstallDir = if ($env:OptiX_INSTALL_DIR) { $env:OptiX_INSTALL_DIR } else { "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.0.0" }
if (-not (Test-Path (Join-Path $OptiXInstallDir "include\optix.h"))) {
    Write-Host "Error: OptiX SDK not found at $OptiXInstallDir" -ForegroundColor Red
    Write-Host "Please set OptiX_INSTALL_DIR to your OptiX installation path" -ForegroundColor Yellow
    exit 1
}

Write-Host "Using OptiX SDK at: $OptiXInstallDir" -ForegroundColor Green

# Check for conda CUDA toolkit
$condaNvcc = Join-Path $env:CONDA_PREFIX "Library\bin\nvcc.exe"
if (-not (Test-Path $condaNvcc)) {
    Write-Host "Error: conda CUDA toolkit not found at $condaNvcc" -ForegroundColor Red
    Write-Host "Please ensure cuda-toolkit is installed in the conda environment:" -ForegroundColor Yellow
    Write-Host "  conda install -c nvidia cuda-toolkit=12.8" -ForegroundColor Yellow
    exit 1
}

Write-Host "Using conda CUDA toolkit: $condaNvcc" -ForegroundColor Green

# Set up library and include paths for conda
$env:CMAKE_PREFIX_PATH = $env:CONDA_PREFIX
$env:CUDA_PATH = $env:CONDA_PREFIX
$env:OptiX_INSTALL_DIR = $OptiXInstallDir

# Download tinyobjloader if missing
$TinyObjDir = Join-Path $ProjectDir "external\tinyobjloader"
if (-not (Test-Path (Join-Path $TinyObjDir "tiny_obj_loader.h")) -or -not (Test-Path (Join-Path $TinyObjDir "tiny_obj_loader.cc"))) {
    Write-Host "Downloading tinyobjloader..." -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $TinyObjDir | Out-Null
    
    $headerUrl = "https://raw.githubusercontent.com/tinyobjloader/tinyobjloader/release/tiny_obj_loader.h"
    $sourceUrl = "https://raw.githubusercontent.com/tinyobjloader/tinyobjloader/release/tiny_obj_loader.cc"
    
    try {
        Invoke-WebRequest -Uri $headerUrl -OutFile (Join-Path $TinyObjDir "tiny_obj_loader.h") -UseBasicParsing
        Invoke-WebRequest -Uri $sourceUrl -OutFile (Join-Path $TinyObjDir "tiny_obj_loader.cc") -UseBasicParsing
        Write-Host "tinyobjloader downloaded to $TinyObjDir" -ForegroundColor Green
    } catch {
        Write-Host "Error downloading tinyobjloader: $_" -ForegroundColor Red
        exit 1
    }
}

# Create build directory
New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
Push-Location $BuildDir

Write-Host ""
Write-Host "Running CMake..." -ForegroundColor Cyan
cmake -S $ProjectDir -B $BuildDir -DCMAKE_BUILD_TYPE=$BuildType -DOptiX_INSTALL_DIR=$OptiXInstallDir

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
Write-Host "Run examples: $(Join-Path $BuildDir "bin\$BuildType\raytracer.exe")" -ForegroundColor Green
Write-Host ""

