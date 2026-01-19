# Project Structure

This document provides an overview of the project structure and the purpose of each file and directory.

```
capstone-2/
├── data/                           # Dataset files
│   └── dataset.csv                 # Main dataset (downloaded from Kaggle)
│
├── docs/                           # Documentation files
│   ├── business_impact.md          # Business impact analysis with ROI calculations
│   ├── evaluation_review.md        # Review against ML Zoomcamp evaluation criteria
│   ├── gcp_deployment.md           # Detailed GCP deployment guide
│   ├── project_structure.md        # This file - project structure overview
│   └── project_summary.md          # High-level project summary
│
├── k8s/                            # Kubernetes deployment files
│   ├── deployment.yaml             # Deployment configuration
│   ├── hpa.yaml                    # Horizontal Pod Autoscaler configuration
│   └── service.yaml                # Service configuration for external access
│
├── logs/                           # Log files (created during execution)
│   ├── prediction.log              # Logs from prediction service
│   └── training.log                # Logs from model training
│
├── models/                         # Saved model files
│   ├── baseline_logistic_regression.joblib  # Baseline model
│   ├── tuned_random_forest.joblib           # Tuned Random Forest model
│   ├── tuned_gradient_boosting.joblib       # Tuned Gradient Boosting model
│   ├── tuned_xgboost.joblib                 # Tuned XGBoost model
│   ├── best_model_*.joblib                  # Best performing model
│   ├── model_metadata.json                  # Model metadata
│   ├── feature_importance.png               # Feature importance visualization
│   ├── model_comparison.png                 # Model comparison visualization
│   └── *_confusion_matrix.png               # Confusion matrices for each model
│
├── notebook/                       # Jupyter notebooks for exploration and modeling
│   ├── student_dropout_analysis.ipynb       # Main analysis notebook
│   └── student_dropout_analysis_template.py # Template for notebook generation
│
├── scripts/                        # Python scripts for training and prediction
│   ├── convert_to_notebook.py      # Script to convert template to Jupyter notebook
│   ├── download_data.py            # Script to download dataset from Kaggle
│   ├── gcp_setup.sh                # Script for setting up GCP deployment
│   ├── predict.py                  # Script for serving predictions via API
│   ├── run_workflow.sh             # Script to run the entire workflow
│   ├── setup_environment.sh        # Script to set up virtual environment
│   ├── test_prediction.py          # Script for testing the prediction service
│   └── train.py                    # Script for training the final model
│
├── .venv/                          # Virtual environment (created during setup)
├── Dockerfile                      # Dockerfile for containerization
├── README.md                       # Project documentation
└── requirements.txt                # Python dependencies
```

## Key Components

### Data Management

- `data/`: Contains the dataset and visualizations generated during EDA
- `download_data.py`: Script to download the dataset from Kaggle

### Exploratory Data Analysis

- `notebook/student_dropout_analysis.ipynb`: Jupyter notebook for data exploration and modeling
- `notebook/student_dropout_analysis_template.py`: Template for generating the notebook

### Model Development

- `train.py`: Script for training the final model with proper validation and hyperparameter tuning
- `models/`: Directory containing saved models and evaluation metrics

### Model Deployment

- `predict.py`: Script for serving predictions via a RESTful API
- `test_prediction.py`: Script for testing the prediction service
- `Dockerfile`: Configuration for containerizing the prediction service
- `k8s/`: Kubernetes manifests for deploying to GKE

### Documentation

- `README.md`: Main project documentation
- `docs/`: Detailed documentation on various aspects of the project
- `business_impact.md`: Analysis of the business impact with ROI calculations
- `gcp_deployment.md`: Guide for deploying to Google Cloud Platform

### Utilities

- `setup_environment.sh`: Script to set up the virtual environment
- `convert_to_notebook.py`: Script to convert Python template to Jupyter notebook
- `run_workflow.sh`: Script to run the entire workflow from data download to deployment

## Workflow

The typical workflow for this project is:

1. Set up the environment using `setup_environment.sh`
2. Download the dataset using `download_data.py`
3. Explore the data and develop models in `student_dropout_analysis.ipynb`
4. Train the final model using `train.py`
5. Serve predictions using `predict.py`
6. Test the prediction service using `test_prediction.py`
7. Deploy to GCP using `gcp_setup.sh`

Alternatively, you can run the entire workflow using `run_workflow.sh`.