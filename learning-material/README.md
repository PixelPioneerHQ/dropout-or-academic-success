# Student Dropout Prediction

## Problem Description

Educational institutions face significant challenges with student dropout rates. When students drop out, it represents:
- Lost tuition revenue for the institution
- Inefficient use of educational resources
- Negative impact on institutional performance metrics
- Potentially negative outcomes for the students themselves

This project aims to develop a machine learning model that can predict which students are at risk of dropping out based on information available at enrollment time. By identifying at-risk students early, institutions can implement targeted intervention strategies to improve retention rates.

## Business Impact

Early identification of at-risk students allows institutions to:
1. **Reduce Revenue Loss**: Prevent tuition revenue loss from dropouts
2. **Optimize Resource Allocation**: Direct support resources to students who need them most
3. **Improve Institutional Metrics**: Enhance graduation rates and institutional rankings
4. **Better Student Outcomes**: Help more students successfully complete their education

## Dataset

This project uses the "Dropout or Academic Success" dataset from Kaggle, which contains information about students' academic paths, demographics, and socio-economic factors. The dataset allows for a three-category classification task:
- Dropout: Students who leave without completing their degree
- Enrolled: Students who are still pursuing their degree
- Graduate: Students who successfully complete their degree

## Project Structure

```
capstone-2/
├── data/               # Dataset files
├── notebook/           # Jupyter notebooks for exploration and modeling
├── models/             # Saved model files
├── scripts/            # Python scripts for training and prediction
│   ├── train.py        # Script for training the final model
│   └── predict.py      # Script for serving predictions via API
├── k8s/                # Kubernetes deployment files
│   ├── deployment.yaml # Deployment configuration
│   ├── service.yaml    # Service configuration
│   └── hpa.yaml        # Horizontal Pod Autoscaler configuration
└── README.md           # Project documentation
```

## Model Approach

The project will:
1. Explore and preprocess the dataset
2. Develop a proper validation strategy for multi-class classification
3. Train multiple model types (logistic regression, decision trees, random forest, gradient boosting)
4. Implement systematic hyperparameter tuning
5. Select the best performing model based on appropriate metrics
6. Deploy the model as a web service

## Deployment

The final model will be:
1. Containerized using Docker
2. Deployed to Google Kubernetes Engine (GKE)
3. Exposed as a RESTful API for real-time predictions

## Usage Instructions

Detailed usage instructions will be provided upon completion of the project.