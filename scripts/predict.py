#!/usr/bin/env python
"""
Prediction service for Student Dropout Prediction model.
This script creates a Flask API that serves predictions from our trained model.
"""

import os
import pandas as pd
import numpy as np
import joblib
import json
from flask import Flask, request, jsonify
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("../logs/prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("../logs", exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Global variable to store the model
model = None

def load_model(model_path):
    """
    Load the trained model from disk
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        object: Loaded model
    """
    logger.info(f"Loading model from {model_path}")
    try:
        loaded_model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        return loaded_model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

@app.before_first_request
def initialize():
    """Initialize the model before the first request"""
    global model
    model_path = os.environ.get('MODEL_PATH', '../models/model.joblib')
    model = load_model(model_path)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expects a JSON with student data in the request body
    Returns prediction and probability
    """
    global model
    
    # Load model if not already loaded
    if model is None:
        model_path = os.environ.get('MODEL_PATH', '../models/model.joblib')
        model = load_model(model_path)
    
    # Get request data
    try:
        data = request.get_json()
        logger.info("Received prediction request")
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        
        # Create response
        classes = model.classes_.tolist()
        proba_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        
        response = {
            'prediction': prediction,
            'probabilities': proba_dict,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: {prediction}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/metadata', methods=['GET'])
def metadata():
    """
    Metadata endpoint
    
    Returns information about the model and its features
    """
    global model
    
    # Load model if not already loaded
    if model is None:
        model_path = os.environ.get('MODEL_PATH', '../models/model.joblib')
        model = load_model(model_path)
    
    try:
        # Get model metadata
        metadata = {
            'model_type': type(model.named_steps['classifier']).__name__,
            'classes': model.classes_.tolist(),
            'feature_preprocessing': {
                'numerical_features': model.named_steps['preprocessor'].transformers_[0][2],
                'categorical_features': model.named_steps['preprocessor'].transformers_[1][2]
            },
            'version': os.environ.get('MODEL_VERSION', '1.0.0')
        }
        
        # Try to load metrics if available
        metrics_path = os.path.join(os.path.dirname(os.environ.get('MODEL_PATH', '../models/model.joblib')), 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            metadata['performance_metrics'] = metrics
        
        return jsonify(metadata)
    
    except Exception as e:
        logger.error(f"Error getting metadata: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/predict/batch', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint
    
    Expects a JSON with an array of student data in the request body
    Returns predictions and probabilities for each input
    """
    global model
    
    # Load model if not already loaded
    if model is None:
        model_path = os.environ.get('MODEL_PATH', '../models/model.joblib')
        model = load_model(model_path)
    
    # Get request data
    try:
        data = request.get_json()
        logger.info(f"Received batch prediction request with {len(data)} samples")
        
        # Convert to DataFrame
        input_data = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(input_data).tolist()
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)
        
        # Create response
        classes = model.classes_.tolist()
        results = []
        
        for i, pred in enumerate(predictions):
            proba_dict = {cls: float(prob) for cls, prob in zip(classes, probabilities[i])}
            results.append({
                'prediction': pred,
                'probabilities': proba_dict
            })
        
        response = {
            'results': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction completed for {len(results)} samples")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Error making batch prediction: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/example', methods=['GET'])
def example_input():
    """
    Example input endpoint
    
    Returns an example of the expected input format
    """
    global model
    
    # Load model if not already loaded
    if model is None:
        model_path = os.environ.get('MODEL_PATH', '../models/model.joblib')
        model = load_model(model_path)
    
    try:
        # Get feature names from the model
        numerical_features = model.named_steps['preprocessor'].transformers_[0][2]
        categorical_features = model.named_steps['preprocessor'].transformers_[1][2]
        
        # Create example input
        example = {}
        
        # Add numerical features with example values
        for feature in numerical_features:
            example[feature] = 0.0
        
        # Add categorical features with example values
        for feature in categorical_features:
            example[feature] = "example_value"
        
        return jsonify({
            'example_input': example,
            'note': 'Replace the example values with actual student data'
        })
    
    except Exception as e:
        logger.error(f"Error creating example input: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    # Load environment variables
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Initialize model
    model_path = os.environ.get('MODEL_PATH', '../models/model.joblib')
    model = load_model(model_path)
    
    # Run the app
    logger.info(f"Starting prediction service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)