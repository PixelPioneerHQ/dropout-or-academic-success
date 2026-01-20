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
from flask_restx import Api, Resource, fields, Namespace
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("./logs/prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
os.makedirs("./logs", exist_ok=True)

# Initialize Flask app
app = Flask(__name__)

# Initialize Flask-RESTX API
api = Api(
    app,
    title='Student Dropout Prediction API',
    version='1.0.0',
    description='Machine Learning API for predicting student dropout, enrollment, or graduation outcomes',
    doc='/docs/',  # Swagger UI available at /docs/
)

# Create namespaces for API organization
ns_health = api.namespace('health', description='Service health operations')
ns_predict = api.namespace('predict', description='ML prediction operations')
ns_metadata = api.namespace('metadata', description='Model information and metadata')

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
        model_path = os.environ.get('MODEL_PATH', './models/best_model_gradient_boosting.joblib')
        model, label_encoder = load_model(model_path)

# Define API models for documentation
student_input_model = api.model('StudentInput', {
    'Admission grade': fields.Float(required=True, description='Student admission grade', example=127.3),
    'Age at enrollment': fields.Integer(required=True, description='Age when student enrolled', example=20),
    'Application mode': fields.Integer(required=True, description='Application mode code', example=17),
    'Application order': fields.Integer(required=True, description='Order of application', example=5),
    'Course': fields.Integer(required=True, description='Course code', example=171),
    'Curricular units 1st sem (approved)': fields.Integer(required=True, description='Approved units in 1st semester', example=0),
    'Curricular units 1st sem (credited)': fields.Integer(required=True, description='Credited units in 1st semester', example=0),
    'Curricular units 1st sem (enrolled)': fields.Integer(required=True, description='Enrolled units in 1st semester', example=0),
    'Curricular units 1st sem (evaluations)': fields.Integer(required=True, description='Evaluations in 1st semester', example=0),
    'Curricular units 1st sem (grade)': fields.Float(required=True, description='Grade in 1st semester', example=0.0),
    'Curricular units 1st sem (without evaluations)': fields.Integer(required=True, description='Units without evaluations in 1st semester', example=0),
    'Curricular units 2nd sem (approved)': fields.Integer(required=True, description='Approved units in 2nd semester', example=0),
    'Curricular units 2nd sem (credited)': fields.Integer(required=True, description='Credited units in 2nd semester', example=0),
    'Curricular units 2nd sem (enrolled)': fields.Integer(required=True, description='Enrolled units in 2nd semester', example=0),
    'Curricular units 2nd sem (evaluations)': fields.Integer(required=True, description='Evaluations in 2nd semester', example=0),
    'Curricular units 2nd sem (grade)': fields.Float(required=True, description='Grade in 2nd semester', example=0.0),
    'Curricular units 2nd sem (without evaluations)': fields.Integer(required=True, description='Units without evaluations in 2nd semester', example=0),
    'Daytime/evening attendance': fields.Integer(required=True, description='Attendance type (0: Evening, 1: Daytime)', example=1),
    'Debtor': fields.Integer(required=True, description='Debtor status (0: No, 1: Yes)', example=0),
    'Displaced': fields.Integer(required=True, description='Displaced student (0: No, 1: Yes)', example=1),
    'Educational special needs': fields.Integer(required=True, description='Special educational needs (0: No, 1: Yes)', example=0),
    'Father\'s occupation': fields.Integer(required=True, description='Father occupation code', example=9),
    'Father\'s qualification': fields.Integer(required=True, description='Father qualification code', example=12),
    'GDP': fields.Float(required=True, description='GDP when enrolled', example=1.74),
    'Gender': fields.Integer(required=True, description='Student gender (0: Female, 1: Male)', example=1),
    'Inflation rate': fields.Float(required=True, description='Inflation rate when enrolled', example=1.4),
    'International': fields.Integer(required=True, description='International student (0: No, 1: Yes)', example=0),
    'Marital status': fields.Integer(required=True, description='Marital status code', example=1),
    'Mother\'s occupation': fields.Integer(required=True, description='Mother occupation code', example=5),
    'Mother\'s qualification': fields.Integer(required=True, description='Mother qualification code', example=19),
    'Nacionality': fields.Integer(required=True, description='Nationality code', example=1),
    'Previous qualification': fields.Integer(required=True, description='Previous qualification code', example=1),
    'Previous qualification (grade)': fields.Float(required=True, description='Previous qualification grade', example=122.0),
    'Scholarship holder': fields.Integer(required=True, description='Scholarship holder (0: No, 1: Yes)', example=0),
    'Tuition fees up to date': fields.Integer(required=True, description='Tuition fees status (0: No, 1: Yes)', example=1),
    'Unemployment rate': fields.Float(required=True, description='Unemployment rate when enrolled', example=10.8)
})

prediction_response_model = api.model('PredictionResponse', {
    'prediction': fields.String(description='Predicted outcome', example='Graduate'),
    'probabilities': fields.Raw(description='Prediction probabilities for each class', example={
        'Dropout': 0.15,
        'Enrolled': 0.25,
        'Graduate': 0.60
    }),
    'timestamp': fields.String(description='Prediction timestamp', example='2024-01-20T08:00:00')
})

batch_prediction_response_model = api.model('BatchPredictionResponse', {
    'results': fields.List(fields.Nested(prediction_response_model), description='List of predictions'),
    'count': fields.Integer(description='Number of predictions', example=3),
    'timestamp': fields.String(description='Batch prediction timestamp', example='2024-01-20T08:00:00')
})

metadata_response_model = api.model('MetadataResponse', {
    'model_type': fields.String(description='Type of ML model', example='GradientBoostingClassifier'),
    'classes': fields.List(fields.String, description='Available prediction classes', example=['Dropout', 'Enrolled', 'Graduate']),
    'feature_preprocessing': fields.Raw(description='Feature preprocessing information'),
    'version': fields.String(description='Model version', example='1.0.0'),
    'performance_metrics': fields.Raw(description='Model performance metrics')
})

health_response_model = api.model('HealthResponse', {
    'status': fields.String(description='Service health status', example='healthy')
})

error_model = api.model('Error', {
    'error': fields.String(description='Error message', example='Invalid input data')
})

@ns_health.route('/')
class HealthCheck(Resource):
    @ns_health.doc('health_check')
    @ns_health.marshal_with(health_response_model)
    def get(self):
        """Health check endpoint
        
        Returns the current health status of the prediction service.
        Use this endpoint to verify that the service is running and accessible.
        """
        return {'status': 'healthy'}

@ns_predict.route('/')
class PredictStudent(Resource):
    @ns_predict.doc('predict_student_outcome')
    @ns_predict.expect(student_input_model)
    @ns_predict.marshal_with(prediction_response_model)
    @ns_predict.response(400, 'Invalid input data', error_model)
    def post(self):
        """Predict student dropout/enrollment/graduation outcome
        
        Make a prediction for a single student based on their academic and demographic data.
        The model uses Gradient Boosting with 76.6% accuracy to predict whether a student will:
        - Dropout: Leave the program before completion
        - Enrolled: Continue in the program
        - Graduate: Successfully complete the program
        
        Returns both the predicted class and probability scores for all classes.
        """
        global model, label_encoder
        
        # Initialize model and label encoder if not already loaded
        initialize()
        
        # Get request data
        try:
            data = api.payload
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
            return response
        
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            api.abort(400, f"Error making prediction: {str(e)}")

@ns_metadata.route('/')
class ModelMetadata(Resource):
    @ns_metadata.doc('get_model_metadata')
    @ns_metadata.marshal_with(metadata_response_model)
    @ns_metadata.response(400, 'Error retrieving metadata', error_model)
    def get(self):
        """Get model metadata and information
        
        Returns detailed information about the machine learning model including:
        - Model type and version
        - Available prediction classes
        - Feature preprocessing details
        - Performance metrics (if available)
        
        This endpoint is useful for understanding the model structure and capabilities.
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
            
            return metadata
        
        except Exception as e:
            logger.error(f"Error getting metadata: {e}")
            api.abort(400, f"Error retrieving metadata: {str(e)}")

@ns_predict.route('/batch')
class BatchPredictStudents(Resource):
    @ns_predict.doc('batch_predict_student_outcomes')
    @ns_predict.expect([student_input_model])
    @ns_predict.marshal_with(batch_prediction_response_model)
    @ns_predict.response(400, 'Invalid input data', error_model)
    def post(self):
        """Predict outcomes for multiple students in batch
        
        Make predictions for multiple students at once. This endpoint is more efficient
        when you need to predict outcomes for many students simultaneously.
        
        Expects an array of student data objects in the request body.
        Returns predictions and probabilities for each student in the batch.
        
        Maximum recommended batch size is 1000 students for optimal performance.
        """
        global model, label_encoder
        
        # Initialize model and label encoder if not already loaded
        initialize()
        
        # Get request data
        try:
            data = api.payload
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
            return response
        
        except Exception as e:
            logger.error(f"Error making batch prediction: {e}")
            api.abort(400, f"Error making batch prediction: {str(e)}")

example_response_model = api.model('ExampleResponse', {
    'example_input': fields.Nested(student_input_model, description='Example student data for testing'),
    'note': fields.String(description='Description of the example data', example='Test student data')
})

@ns_predict.route('/example')
class ExampleInput(Resource):
    @ns_predict.doc('get_example_input')
    @ns_predict.marshal_with(example_response_model)
    @ns_predict.response(400, 'Error creating example', error_model)
    def get(self):
        """Get example input data for testing
        
        Returns a properly formatted example of student data that can be used
        to test the prediction endpoints. This example includes realistic values
        for all required fields.
        
        You can copy this example data and modify the values to test different
        scenarios with the /predict endpoint.
        """
        global model, label_encoder
        
        # Initialize model and label encoder if not already loaded
        initialize()
        
        try:
            # Create example input with realistic student data
            example = {
                "Admission grade": 127.3,
                "Age at enrollment": 20,
                "Application mode": 17,
                "Application order": 5,
                "Course": 171,
                "Curricular units 1st sem (approved)": 0,
                "Curricular units 1st sem (credited)": 0,
                "Curricular units 1st sem (enrolled)": 0,
                "Curricular units 1st sem (evaluations)": 0,
                "Curricular units 1st sem (grade)": 0,
                "Curricular units 1st sem (without evaluations)": 0,
                "Curricular units 2nd sem (approved)": 0,
                "Curricular units 2nd sem (credited)": 0,
                "Curricular units 2nd sem (enrolled)": 0,
                "Curricular units 2nd sem (evaluations)": 0,
                "Curricular units 2nd sem (grade)": 0,
                "Curricular units 2nd sem (without evaluations)": 0,
                "Daytime/evening attendance": 1,
                "Debtor": 0,
                "Displaced": 1,
                "Educational special needs": 0,
                "Father's occupation": 9,
                "Father's qualification": 12,
                "GDP": 1.74,
                "Gender": 1,
                "Inflation rate": 1.4,
                "International": 0,
                "Marital status": 1,
                "Mother's occupation": 5,
                "Mother's qualification": 19,
                "Nacionality": 1,
                "Previous qualification": 1,
                "Previous qualification (grade)": 122,
                "Scholarship holder": 0,
                "Tuition fees up to date": 1,
                "Unemployment rate": 10.8
            }
            
            return {
                'example_input': example,
                'note': 'Test student data - modify values as needed for testing'
            }
        
        except Exception as e:
            logger.error(f"Error creating example input: {e}")
            api.abort(400, f"Error creating example input: {str(e)}")

if __name__ == "__main__":
    # Load environment variables
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    # Initialize model and label encoder
    initialize()
    
    # Run the app
    logger.info(f"Starting prediction service on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)