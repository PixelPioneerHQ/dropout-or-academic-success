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

# Global variables to store the model and label encoder
model = None
label_encoder = None

def load_model(model_path):
    """
    Load the trained model and label encoder from disk
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        tuple: (loaded_model, label_encoder)
    """
    logger.info(f"Loading model from {model_path}")
    try:
        loaded_model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        # Load label encoder from the same directory
        model_dir = os.path.dirname(model_path)
        label_encoder_path = os.path.join(model_dir, 'label_encoder.joblib')
        
        if os.path.exists(label_encoder_path):
            loaded_label_encoder = joblib.load(label_encoder_path)
            logger.info("Label encoder loaded successfully")
        else:
            logger.warning("Label encoder not found, predictions will be numeric")
            loaded_label_encoder = None
            
        return loaded_model, loaded_label_encoder
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def initialize():
    """Initialize the model and label encoder"""
    global model, label_encoder
    if model is None or label_encoder is None:
        # Load the best performing model from research (Tuned Gradient Boosting - 0.7804 accuracy)
        model_path = os.environ.get('MODEL_PATH', '../models/best_model_gradient_boosting.joblib')
        model, label_encoder = load_model(model_path)

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
    global model, label_encoder
    
    # Initialize model and label encoder if not already loaded
    initialize()
    
    # Get request data
    try:
        data = request.get_json()
        logger.info("Received prediction request")
        
        # Convert to DataFrame
        input_data = pd.DataFrame([data])
        
        # Make prediction (returns numeric label)
        prediction_numeric = model.predict(input_data)[0]
        
        # Convert to original label if label encoder is available
        if label_encoder is not None:
            prediction = label_encoder.inverse_transform([prediction_numeric])[0]
            classes = label_encoder.classes_.tolist()
        else:
            prediction = prediction_numeric
            classes = [0, 1, 2]  # Default numeric classes
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)[0]
        
        # Create response
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
    global model, label_encoder
    
    # Initialize model and label encoder if not already loaded
    initialize()
    
    try:
        # Get model metadata
        classes = label_encoder.classes_.tolist() if label_encoder else [0, 1, 2]
        metadata = {
            'model_type': type(model.named_steps['classifier']).__name__,
            'classes': classes,
            'feature_preprocessing': {
                'numerical_features': model.named_steps['preprocessor'].transformers_[0][2],
                'categorical_features': model.named_steps['preprocessor'].transformers_[1][2]
            },
            'version': os.environ.get('MODEL_VERSION', '1.0.0')
        }
        
        # Try to load metrics if available
        metrics_path = os.path.join(os.path.dirname(os.environ.get('MODEL_PATH', '../models/best_model_gradient_boosting.joblib')), 'metrics.json')
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
    global model, label_encoder
    
    # Initialize model and label encoder if not already loaded
    initialize()
    
    # Get request data
    try:
        data = request.get_json()
        logger.info(f"Received batch prediction request with {len(data)} samples")
        
        # Convert to DataFrame
        input_data = pd.DataFrame(data)
        
        # Make predictions (returns numeric labels)
        predictions_numeric = model.predict(input_data)
        
        # Convert to original labels if label encoder is available
        if label_encoder is not None:
            predictions = label_encoder.inverse_transform(predictions_numeric).tolist()
            classes = label_encoder.classes_.tolist()
        else:
            predictions = predictions_numeric.tolist()
            classes = [0, 1, 2]  # Default numeric classes
        
        # Get prediction probabilities
        probabilities = model.predict_proba(input_data)
        
        # Create response
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
    global model, label_encoder
    
    # Initialize model and label encoder if not already loaded
    initialize()
    
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
    
    # Initialize model and label encoder
    initialize()
    
    # Run the app
    logger.info(f"Starting prediction service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)