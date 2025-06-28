# PowerShell script to run background removal with virtual environment
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path $scriptDir "venv\Scripts\Activate.ps1"
$pythonScript = Join-Path $scriptDir "bg-remove.py"

# Activate virtual environment and run the script
& $venvPath
python $pythonScript $args
