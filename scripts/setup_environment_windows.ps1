# Windows PowerShell script to set up virtual environment for Student Dropout Prediction
# Run this script from the root directory of the project

Write-Host "Setting up Python virtual environment for Student Dropout Prediction..." -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.9+ from https://python.org" -ForegroundColor Yellow
    exit 1
}

# Create virtual environment
if (Test-Path ".venv") {
    Write-Host "Virtual environment already exists. Removing old one..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force ".venv"
}

Write-Host "Creating virtual environment..." -ForegroundColor Blue
python -m venv .venv

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Blue
& ".venv\Scripts\Activate.ps1"

# Upgrade pip
Write-Host "Upgrading pip..." -ForegroundColor Blue
python -m pip install --upgrade pip

# Install requirements
Write-Host "Installing requirements..." -ForegroundColor Blue
pip install -r requirements.txt

Write-Host "" -ForegroundColor Green
Write-Host "✅ Setup complete!" -ForegroundColor Green
Write-Host "" -ForegroundColor Green
Write-Host "To activate the virtual environment in future sessions, run:" -ForegroundColor Yellow
Write-Host ".venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Green
Write-Host "To deactivate the virtual environment, run:" -ForegroundColor Yellow
Write-Host "deactivate" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Green
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Download the dataset: python scripts/download_data.py" -ForegroundColor Cyan
Write-Host "2. Open Jupyter: jupyter notebook" -ForegroundColor Cyan
Write-Host "3. Open the .ipynb files in the notebook/ directory" -ForegroundColor Cyan