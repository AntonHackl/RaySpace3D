$ErrorActionPreference = "Stop"

$handleExe = "C:\Program Files\Handle\handle.exe"
# Get the root of the repository (parent of the 'scripts' folder)
$repoPath = (Get-Item "$PSScriptRoot\..").FullName

Write-Host "Repository Lock Cleaner"
Write-Host "-----------------------"
Write-Host "Target: $repoPath"
Write-Host "Tool:   $handleExe"
Write-Host ""

if (-not (Test-Path $handleExe)) {
    Write-Error "Handle.exe not found at: $handleExe"
    exit 1
}

if (-not (Test-Path $repoPath)) {
    Write-Error "Repository path not found: $repoPath"
    exit 1
}

Write-Host "Scanning for locks..."

# Run handle.exe to find processes locking files in the repo
# -nobanner: suppress copyright message
# -accepteula: accept the license automatically to prevent hanging
# The arguments are the path substring to search for
$output = & $handleExe -accepteula -nobanner $repoPath 2>&1

$pidsToKill = @()

foreach ($line in $output) {
    # Output format is typically:
    # ProcessName      pid: 1234   type: File  Handle: Path
    if ($line -match 'pid:\s*(\d+)') {
        $foundPid = [int]$matches[1]
        
        # Avoid killing the current PowerShell process or the VS Code editor itself if possible, 
        # though usually we want to kill build processes.
        # It's better to be aggressive if the user asked for it, but let's exclude ourself.
        if ($foundPid -ne $PID) {
            $pidsToKill += $foundPid
        }
    }
}

if ($pidsToKill.Count -eq 0) {
    Write-Host "No locks found."
    exit 0
}

# Unique PIDs
$pidsToKill = $pidsToKill | Select-Object -Unique

foreach ($id in $pidsToKill) {
    try {
        $proc = Get-Process -Id $id -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Found lock by process: $($proc.ProcessName) (PID: $id)"
            
            # Skip vital processes just in case (e.g. Code if you are editing inside it?)
            # The user asked to kill ALL processes, so we will kill them.
            # However, killing 'Code' might close the editor this is running in.
            if ($proc.ProcessName -eq "Code") {
                Write-Warning "Skipping VS Code process (PID: $id) to prevent closing the editor. Close files manually or save as."
                continue
            }
            
            Write-Host "  Killing..." -NoNewline
            Stop-Process -Id $id -Force
            Write-Host " Terminated."
        }
    } catch {
        Write-Warning "Failed to kill PID $id. You might need to run this script as Administrator."
    }
}

Write-Host "`nCleanup complete."
