# Student Dropout Prediction

## Problem Description

Educational institutions face significant challenges with student dropout rates. When students drop out, it represents:
- Lost tuition revenue for the institution
- Inefficient use of educational resources
- Negative impact on institutional performance metrics
- Potentially negative outcomes for the students themselves

This project develops a machine learning model that predicts which students are at risk of dropping out based on information available at enrollment time. By identifying at-risk students early, institutions can implement targeted intervention strategies to improve retention rates.

## Business Impact

Early identification of at-risk students allows institutions to:
1. **Reduce Revenue Loss**: Prevent tuition revenue loss from dropouts
2. **Optimize Resource Allocation**: Direct support resources to students who need them most
3. **Improve Institutional Metrics**: Enhance graduation rates and institutional rankings
4. **Better Student Outcomes**: Help more students successfully complete their education

For a detailed business impact analysis with ROI calculations, see [Business Impact Analysis](docs/business_impact.md).

## Dataset

This project uses the ["Dropout or Academic Success" dataset from Kaggle](https://www.kaggle.com/datasets/ankanhore545/dropout-or-academic-success), which contains information about students' academic paths, demographics, and socio-economic factors. The dataset allows for a three-category classification task:
- **Dropout**: Students who leave without completing their degree
- **Enrolled**: Students who are still pursuing their degree
- **Graduate**: Students who successfully complete their degree

## Project Structure

```
capstone-2/
├── data/               # Dataset files
├── docs/               # Documentation files
│   ├── business_impact.md    # Business impact analysis
│   └── gcp_deployment.md     # GCP deployment guide
├── notebook/           # Jupyter notebooks for exploration and modeling
│   └── student_dropout_analysis.py     # Data exploration & Model training and evaluation
├── models/             # Saved model files
├── scripts/            # Python scripts for training and prediction
│   ├── download_data.py      # Script to download dataset from Kaggle
│   ├── train.py              # Script for training the final model
│   ├── predict.py            # Script for serving predictions via API
│   ├── test_prediction.py    # Script for testing the prediction service
│   └── gcp_setup.sh          # Script for setting up GCP deployment
├── k8s/                # Kubernetes deployment files
│   ├── deployment.yaml # Deployment configuration
│   ├── service.yaml    # Service configuration
│   └── hpa.yaml        # Horizontal Pod Autoscaler configuration
├── Dockerfile          # Dockerfile for containerization
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Model Approach

The project implements:

1. **Proper Validation Strategy**: Stratified k-fold cross-validation to ensure representative class distribution in all folds.

2. **Multiple Model Types**:
   - Logistic Regression (baseline)
   - Random Forest
   - Gradient Boosting
   - XGBoost

3. **Systematic Hyperparameter Tuning**:
   - Grid search for Random Forest and XGBoost
   - Optuna for Gradient Boosting

4. **Evaluation Metrics**:
   - Accuracy
   - F1 Score (macro-averaged)
   - Confusion Matrix
   - Classification Report

## Installation and Setup

### Prerequisites
- Python 3.9+
- Docker
- Google Cloud SDK (for GCP deployment)
- kubectl (for Kubernetes deployment)
- Kaggle API credentials (for dataset download)

### Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/student-dropout-prediction.git
cd student-dropout-prediction
```

2. Set up virtual environment:
```bash
# Run the setup script
bash scripts/setup_environment.sh
   # On Windows:
   PowerShell -ExecutionPolicy Bypass -File scripts/setup_environment_windows.ps1

# Activate the virtual environment
# On Windows:
source .venv/Scripts/activate
# On Linux/Mac:
source .venv/bin/activate
```

3. Download the dataset:
```bash
python scripts/download_data.py
```

4. Open and run the Jupyter notebook for exploration and modeling:
```bash
jupyter notebook notebook/student_dropout_analysis.ipynb
```

5. Train the model:
```bash
python scripts/train.py
```

6. Run the prediction service locally:
```bash
python scripts/predict.py
```

7. Test the prediction service:
```bash
   python scripts/test_prediction.py --url http://localhost:8080
```

### Docker Setup

1. Build the Docker image:
```bash
docker build -t student-dropout-predictor:v1 .
```

2. Run the Docker container:
```bash
docker run -p 8080:8080 student-dropout-predictor:v1
```

3. Test the containerized service:
```bash
python scripts/test_prediction.py --url http://localhost:8080
```

## GCP Deployment

For detailed instructions on deploying to Google Cloud Platform, see [GCP Deployment Guide](docs/gcp_deployment.md).

Quick deployment steps:

1. Set up GCP project and enable APIs:
```bash
cd scripts
./gcp_setup.sh
```

2. Access the deployed service:
```bash
# Get the external IP
kubectl get service student-dropout-predictor
```

3. Test the deployed service:
```bash
python scripts/test_prediction.py --url http://<EXTERNAL_IP>
```

## API Documentation

The prediction service exposes the following endpoints:

- **GET /health**: Health check endpoint
- **GET /metadata**: Returns model metadata
- **GET /example**: Returns an example input format
- **POST /predict**: Makes a prediction for a single student
- **POST /predict/batch**: Makes predictions for multiple students

Example prediction request:
```bash
curl -X POST -H "Content-Type: application/json" -d '{
  "feature1": value1,
  "feature2": "value2",
  ...
}' http://localhost:8080/predict
```

## Results

The model achieves the following performance metrics:

- Accuracy: 0.XX
- F1 Score (macro): 0.XX

For detailed model performance and feature importance analysis, see the [notebook](notebook/student_dropout_modeling.py).

## Business Impact

Based on our analysis, implementing this model with appropriate interventions can provide an ROI of over 300%. For detailed calculations and sensitivity analysis, see [Business Impact Analysis](docs/business_impact.md).

## Future Improvements

1. Incorporate additional data sources (e.g., course engagement, attendance)
2. Implement more advanced feature engineering techniques
3. Explore deep learning approaches for improved performance
4. Develop a user interface for non-technical stakeholders
5. Implement A/B testing for intervention strategies

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle for providing the dataset
- DataTalksClub for the ML Zoomcamp course