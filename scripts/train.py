#!/usr/bin/env python
"""
Training script for Student Dropout Prediction model.
This script implements the best model from our experiments with proper
separation of concerns and modularity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
import argparse
import json
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("../logs", exist_ok=True)
os.makedirs("../models", exist_ok=True)

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def load_data(data_path):
    """
    Load the dataset from the specified path
    
    Args:
        data_path (str): Path to the dataset CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    logger.info(f"Loading data from {data_path}")
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df, target_col):
    """
    Preprocess the data and split into train and test sets
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, preprocessor
    """
    logger.info("Preprocessing data...")
    
    # Split features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    logger.info(f"Categorical columns: {len(categorical_cols)}")
    logger.info(f"Numerical columns: {len(numerical_cols)}")
    
    # Create preprocessing pipelines
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
    
    # Split data into train and test sets using stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info(f"Training set size: {X_train.shape}")
    logger.info(f"Test set size: {X_test.shape}")
    
    # Check class distribution
    logger.info("Class distribution in training set:")
    logger.info(y_train.value_counts(normalize=True))
    
    return X_train, X_test, y_train, y_test, preprocessor

def train_model(X_train, y_train, preprocessor, model_params=None):
    """
    Train the XGBoost model with the given parameters
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        preprocessor (ColumnTransformer): Data preprocessor
        model_params (dict): Model hyperparameters
        
    Returns:
        Pipeline: Trained model pipeline
    """
    logger.info("Training XGBoost model...")
    
    # Default parameters if none provided
    if model_params is None:
        model_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE
        }
    
    # Create and train the pipeline
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(**model_params))
    ])
    
    model.fit(X_train, y_train)
    logger.info("Model training completed")
    
    return model

def evaluate_model(model, X_test, y_test, output_dir):
    """
    Evaluate the model and save evaluation metrics
    
    Args:
        model (Pipeline): Trained model pipeline
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        output_dir (str): Directory to save evaluation results
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score (macro): {f1:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()
    
    # Save metrics to JSON
    metrics = {
        'accuracy': float(accuracy),
        'f1_score': float(f1),
        'classification_report': report
    }
    
    with open(f'{output_dir}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Evaluation metrics saved to {output_dir}/metrics.json")
    
    return metrics

def save_model(model, output_path):
    """
    Save the trained model to disk
    
    Args:
        model (Pipeline): Trained model pipeline
        output_path (str): Path to save the model
        
    Returns:
        str: Path where model was saved
    """
    logger.info(f"Saving model to {output_path}")
    joblib.dump(model, output_path)
    logger.info("Model saved successfully")
    return output_path

def main(data_path, target_col, output_dir, model_params=None):
    """
    Main training function
    
    Args:
        data_path (str): Path to the dataset
        target_col (str): Name of the target column
        output_dir (str): Directory to save model and metrics
        model_params (dict): Model hyperparameters
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    df = load_data(data_path)
    
    # Preprocess data
    X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df, target_col)
    
    # Train model
    model = train_model(X_train, y_train, preprocessor, model_params)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test, output_dir)
    
    # Save model
    model_path = os.path.join(output_dir, 'model.joblib')
    save_model(model, model_path)
    
    # Save feature importance if available
    try:
        # Get feature names after preprocessing
        preprocessor.fit(X_train)
        feature_names = []
        
        # Get numerical feature names
        if hasattr(preprocessor.named_transformers_['num'], 'get_feature_names_out'):
            num_features = preprocessor.named_transformers_['num'].get_feature_names_out()
            feature_names.extend(num_features)
        else:
            num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
            feature_names.extend(num_cols)
        
        # Get categorical feature names
        if hasattr(preprocessor.named_transformers_['cat'], 'get_feature_names_out'):
            cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out()
            feature_names.extend(cat_features)
        
        # Get feature importance from the model
        feature_importance = model.named_steps['classifier'].feature_importances_
        
        # Create a dataframe of feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': feature_importance
        }).sort_values('Importance', ascending=False)
        
        # Save feature importance
        importance_df.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        logger.info(f"Feature importance saved to {output_dir}/feature_importance.csv")
    except Exception as e:
        logger.warning(f"Could not save feature importance: {e}")
    
    logger.info("Training completed successfully")
    
    return model, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Student Dropout Prediction model')
    parser.add_argument('--data', type=str, default='../data/dataset.csv',
                        help='Path to the dataset CSV file')
    parser.add_argument('--target', type=str, default='Target',
                        help='Name of the target column')
    parser.add_argument('--output', type=str, default='../models',
                        help='Directory to save model and metrics')
    
    args = parser.parse_args()
    
    # Best parameters from hyperparameter tuning
    best_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE
    }
    
    main(args.data, args.target, args.output, best_params)