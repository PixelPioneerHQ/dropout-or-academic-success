#!/bin/bash
# Script to run the entire workflow from data download to model deployment

# Exit on error
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Print section header
section() {
    echo -e "\n${GREEN}=== $1 ===${NC}\n"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
section "Checking prerequisites"

if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 not found. Please install Python 3.${NC}"
    exit 1
fi

if ! command_exists jupyter; then
    echo -e "${YELLOW}Warning: Jupyter not found. Will attempt to install it in the virtual environment.${NC}"
fi

# Set up virtual environment
section "Setting up virtual environment"
echo "Running setup_environment.sh..."
bash setup_environment.sh

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source ../.venv/Scripts/activate
else
    # Linux/Mac
    source ../.venv/bin/activate
fi

# Download dataset
section "Downloading dataset"
echo "Running download_data.py..."
python download_data.py

# Convert template to notebook
section "Converting template to Jupyter notebook"
echo "Running convert_to_notebook.py..."
python convert_to_notebook.py ../notebook/student_dropout_analysis_template.py ../notebook/student_dropout_analysis.ipynb

# Run the notebook
section "Running Jupyter notebook"
echo -e "${YELLOW}Note: This step requires manual interaction.${NC}"
echo "To run the notebook, execute the following command:"
echo "jupyter notebook ../notebook/student_dropout_analysis.ipynb"
echo -e "${YELLOW}Press Enter to continue after you've completed the notebook analysis...${NC}"
read -p ""

# Train the model
section "Training the model"
echo "Running train.py..."
python train.py

# Test the prediction service locally
section "Testing prediction service locally"
echo "Starting prediction service in the background..."
python predict.py &
PREDICT_PID=$!

# Wait for the service to start
echo "Waiting for the service to start..."
sleep 5

# Test the service
echo "Testing the prediction service..."
python test_prediction.py --url http://localhost:8080

# Stop the prediction service
echo "Stopping prediction service..."
kill $PREDICT_PID

# Build Docker image
section "Building Docker image"
echo "Building Docker image..."
cd ..
docker build -t student-dropout-predictor:v1 .

# Test Docker container
section "Testing Docker container"
echo "Running Docker container..."
docker run -d -p 8080:8080 --name student-dropout-predictor student-dropout-predictor:v1

# Wait for the container to start
echo "Waiting for the container to start..."
sleep 5

# Test the container
echo "Testing the Docker container..."
cd scripts
python test_prediction.py --url http://localhost:8080

# Stop and remove the container
echo "Stopping and removing Docker container..."
docker stop student-dropout-predictor
docker rm student-dropout-predictor

# GCP deployment
section "GCP deployment"
echo -e "${YELLOW}Note: GCP deployment requires manual configuration.${NC}"
echo "To deploy to GCP, you need to:"
echo "1. Set up a GCP project"
echo "2. Enable required APIs"
echo "3. Create a GKE cluster"
echo "4. Push the Docker image to Google Container Registry"
echo "5. Deploy the application to GKE"
echo ""
echo "You can use the gcp_setup.sh script for this purpose:"
echo "bash gcp_setup.sh"
echo ""
echo -e "${YELLOW}Do you want to proceed with GCP deployment? (y/n)${NC}"
read -p "" proceed

if [[ $proceed == "y" || $proceed == "Y" ]]; then
    echo "Running gcp_setup.sh..."
    bash gcp_setup.sh
else
    echo "Skipping GCP deployment."
fi

section "Workflow completed"
echo -e "${GREEN}The workflow has been completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Review the model performance in the notebook"
echo "2. Fine-tune the model if needed"
echo "3. Deploy the model to production"
echo ""
echo "For more information, refer to the documentation in the docs/ directory."