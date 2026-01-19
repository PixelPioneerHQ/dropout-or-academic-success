#!/bin/bash
# Script to set up the virtual environment for the project

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up virtual environment for Student Dropout Prediction project...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3 first.${NC}"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv ../.venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source ../.venv/Scripts/activate
else
    # Linux/Mac
    source ../.venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r ../requirements.txt

# Install Jupyter kernel
echo "Installing Jupyter kernel..."
pip install ipykernel
python -m ipykernel install --user --name=student-dropout-env --display-name="Student Dropout Env"

echo -e "${GREEN}Virtual environment setup complete!${NC}"
echo "To activate the virtual environment, run:"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "source .venv/Scripts/activate"
else
    echo "source .venv/bin/activate"
fi
echo "To deactivate the virtual environment, run: deactivate"