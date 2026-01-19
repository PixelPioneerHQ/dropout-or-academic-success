#!/usr/bin/env python
"""
Script to test the prediction service locally or remotely.
"""

import requests
import json
import argparse
import pandas as pd
import numpy as np
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_health(base_url):
    """Test the health endpoint"""
    url = f"{base_url}/health"
    logger.info(f"Testing health endpoint: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        logger.info(f"Health check successful: {response.json()}")
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

def test_metadata(base_url):
    """Test the metadata endpoint"""
    url = f"{base_url}/metadata"
    logger.info(f"Testing metadata endpoint: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        logger.info(f"Metadata retrieved successfully")
        logger.info(json.dumps(response.json(), indent=2))
        return True
    except Exception as e:
        logger.error(f"Metadata retrieval failed: {e}")
        return False

def test_example(base_url):
    """Test the example endpoint"""
    url = f"{base_url}/example"
    logger.info(f"Testing example endpoint: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        logger.info(f"Example retrieved successfully")
        logger.info(json.dumps(response.json(), indent=2))
        return response.json()['example_input']
    except Exception as e:
        logger.error(f"Example retrieval failed: {e}")
        return None

def get_realistic_test_data():
    """Get realistic test data examples"""
    return {
        "high_performing_student": {
            "description": "Student likely to graduate - good grades, engaged",
            "data": {
                "Marital status": 1, "Application mode": 1, "Application order": 1, "Course": 9254,
                "Daytime/evening attendance": 1, "Previous qualification": 1, "Previous qualification (grade)": 160.0,
                "Nacionality": 1, "Mother's qualification": 19, "Father's qualification": 12,
                "Mother's occupation": 5, "Father's occupation": 10, "Admission grade": 150.0,
                "Displaced": 0, "Educational special needs": 0, "Debtor": 0, "Tuition fees up to date": 1,
                "Gender": 1, "Scholarship holder": 1, "Age at enrollment": 18, "International": 0,
                "Curricular units 1st sem (credited)": 0, "Curricular units 1st sem (enrolled)": 6,
                "Curricular units 1st sem (evaluations)": 6, "Curricular units 1st sem (approved)": 6,
                "Curricular units 1st sem (grade)": 14.5, "Curricular units 1st sem (without evaluations)": 0,
                "Curricular units 2nd sem (credited)": 0, "Curricular units 2nd sem (enrolled)": 6,
                "Curricular units 2nd sem (evaluations)": 6, "Curricular units 2nd sem (approved)": 6,
                "Curricular units 2nd sem (grade)": 13.7, "Curricular units 2nd sem (without evaluations)": 0,
                "Unemployment rate": 10.8, "Inflation rate": 1.4, "GDP": 1.74
            }
        },
        "at_risk_student": {
            "description": "Student at risk of dropping out - poor grades, financial issues",
            "data": {
                "Marital status": 2, "Application mode": 17, "Application order": 5, "Course": 171,
                "Daytime/evening attendance": 0, "Previous qualification": 1, "Previous qualification (grade)": 95.0,
                "Nacionality": 1, "Mother's qualification": 2, "Father's qualification": 2,
                "Mother's occupation": 4, "Father's occupation": 4, "Admission grade": 95.5,
                "Displaced": 1, "Educational special needs": 0, "Debtor": 1, "Tuition fees up to date": 0,
                "Gender": 0, "Scholarship holder": 0, "Age at enrollment": 25, "International": 0,
                "Curricular units 1st sem (credited)": 0, "Curricular units 1st sem (enrolled)": 6,
                "Curricular units 1st sem (evaluations)": 4, "Curricular units 1st sem (approved)": 2,
                "Curricular units 1st sem (grade)": 8.5, "Curricular units 1st sem (without evaluations)": 2,
                "Curricular units 2nd sem (credited)": 0, "Curricular units 2nd sem (enrolled)": 6,
                "Curricular units 2nd sem (evaluations)": 2, "Curricular units 2nd sem (approved)": 1,
                "Curricular units 2nd sem (grade)": 6.2, "Curricular units 2nd sem (without evaluations)": 4,
                "Unemployment rate": 13.9, "Inflation rate": -0.3, "GDP": 0.79
            }
        },
        "average_student": {
            "description": "Average student - moderate performance, may be enrolled",
            "data": {
                "Marital status": 1, "Application mode": 15, "Application order": 2, "Course": 8014,
                "Daytime/evening attendance": 1, "Previous qualification": 1, "Previous qualification (grade)": 130.0,
                "Nacionality": 1, "Mother's qualification": 12, "Father's qualification": 10,
                "Mother's occupation": 7, "Father's occupation": 9, "Admission grade": 125.3,
                "Displaced": 0, "Educational special needs": 0, "Debtor": 0, "Tuition fees up to date": 1,
                "Gender": 0, "Scholarship holder": 0, "Age at enrollment": 19, "International": 0,
                "Curricular units 1st sem (credited)": 0, "Curricular units 1st sem (enrolled)": 6,
                "Curricular units 1st sem (evaluations)": 5, "Curricular units 1st sem (approved)": 4,
                "Curricular units 1st sem (grade)": 11.8, "Curricular units 1st sem (without evaluations)": 1,
                "Curricular units 2nd sem (credited)": 0, "Curricular units 2nd sem (enrolled)": 6,
                "Curricular units 2nd sem (evaluations)": 5, "Curricular units 2nd sem (approved)": 4,
                "Curricular units 2nd sem (grade)": 11.2, "Curricular units 2nd sem (without evaluations)": 1,
                "Unemployment rate": 9.4, "Inflation rate": -0.8, "GDP": -3.12
            }
        }
    }

def test_realistic_scenarios(base_url):
    """Test realistic student scenarios"""
    logger.info("Testing realistic student scenarios")
    
    test_data = get_realistic_test_data()
    results = {}
    
    for student_type, student_info in test_data.items():
        logger.info(f"Testing {student_type}: {student_info['description']}")
        
        try:
            response = requests.post(f"{base_url}/predict", json=student_info['data'])
            response.raise_for_status()
            
            result = response.json()
            prediction = result['prediction']
            probabilities = result['probabilities']
            
            logger.info(f"Prediction for {student_type}: {prediction}")
            logger.info(f"Probabilities: {probabilities}")
            
            results[student_type] = result
            
        except Exception as e:
            logger.error(f"Error testing {student_type}: {e}")
    
    return results

def test_prediction(base_url, data=None):
    """Test the prediction endpoint"""
    url = f"{base_url}/predict"
    logger.info(f"Testing prediction endpoint: {url}")
    
    # If no data provided, use realistic example
    if data is None:
        realistic_data = get_realistic_test_data()
        # Use the high performing student as default example
        data = realistic_data['high_performing_student']['data']
        logger.info("Using realistic high-performing student data for testing")
    
    logger.info(f"Sending data: {json.dumps(data, indent=2)}")
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        logger.info(f"Prediction successful")
        logger.info(json.dumps(response.json(), indent=2))
        return response.json()
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None

def test_batch_prediction(base_url, num_samples=3, data=None):
    """Test the batch prediction endpoint with realistic data"""
    url = f"{base_url}/predict/batch"
    logger.info(f"Testing batch prediction endpoint: {url}")
    
    # If no data provided, use realistic student examples
    if data is None:
        realistic_data = get_realistic_test_data()
        data = [student_info['data'] for student_info in realistic_data.values()]
        logger.info("Using realistic student data for batch prediction")
    
    logger.info(f"Sending {len(data)} samples for batch prediction")
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        logger.info(f"Batch prediction successful")
        logger.info(f"Received {len(response.json()['results'])} predictions")
        return response.json()
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return None

def load_test(base_url, num_requests=100, concurrency=10):
    """Perform a simple load test on the prediction endpoint"""
    url = f"{base_url}/predict"
    logger.info(f"Performing load test on {url}")
    logger.info(f"Sending {num_requests} requests with concurrency {concurrency}")
    
    # Get example data
    example = test_example(base_url)
    if not example:
        logger.warning("Using dummy data for load test")
        example = {"feature1": 0, "feature2": "value"}
    
    # Track response times
    response_times = []
    success_count = 0
    error_count = 0
    
    import concurrent.futures
    
    def send_request():
        start_time = time.time()
        try:
            response = requests.post(url, json=example)
            response.raise_for_status()
            end_time = time.time()
            return True, end_time - start_time
        except Exception:
            end_time = time.time()
            return False, end_time - start_time
    
    # Use ThreadPoolExecutor for concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(send_request) for _ in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            success, response_time = future.result()
            response_times.append(response_time)
            if success:
                success_count += 1
            else:
                error_count += 1
    
    # Calculate statistics
    avg_response_time = np.mean(response_times)
    p95_response_time = np.percentile(response_times, 95)
    p99_response_time = np.percentile(response_times, 99)
    
    logger.info(f"Load test completed")
    logger.info(f"Success rate: {success_count}/{num_requests} ({success_count/num_requests*100:.2f}%)")
    logger.info(f"Average response time: {avg_response_time:.4f} seconds")
    logger.info(f"95th percentile response time: {p95_response_time:.4f} seconds")
    logger.info(f"99th percentile response time: {p99_response_time:.4f} seconds")
    
    return {
        "success_rate": success_count/num_requests,
        "avg_response_time": avg_response_time,
        "p95_response_time": p95_response_time,
        "p99_response_time": p99_response_time
    }

def main():
    parser = argparse.ArgumentParser(description='Test the prediction service')
    parser.add_argument('--url', type=str, default='http://localhost:8080',
                        help='Base URL of the prediction service')
    parser.add_argument('--test', type=str, choices=['health', 'metadata', 'example', 'predict', 'batch', 'realistic', 'load', 'all'],
                        default='all', help='Test to run')
    parser.add_argument('--data', type=str, help='JSON file with test data')
    parser.add_argument('--num-requests', type=int, default=100,
                        help='Number of requests for load test')
    parser.add_argument('--concurrency', type=int, default=10,
                        help='Concurrency level for load test')
    
    args = parser.parse_args()
    
    # Load test data if provided
    test_data = None
    if args.data:
        try:
            with open(args.data, 'r') as f:
                test_data = json.load(f)
            logger.info(f"Loaded test data from {args.data}")
        except Exception as e:
            logger.error(f"Error loading test data: {e}")
    
    # Run the specified test
    if args.test == 'health' or args.test == 'all':
        test_health(args.url)
    
    if args.test == 'metadata' or args.test == 'all':
        test_metadata(args.url)
    
    if args.test == 'example' or args.test == 'all':
        test_example(args.url)
    
    if args.test == 'predict' or args.test == 'all':
        test_prediction(args.url, test_data)
    
    if args.test == 'realistic' or args.test == 'all':
        test_realistic_scenarios(args.url)
    
    if args.test == 'batch' or args.test == 'all':
        test_batch_prediction(args.url, 3, test_data)
    
    if args.test == 'load' or args.test == 'all':
        load_test(args.url, args.num_requests, args.concurrency)

if __name__ == "__main__":
    main()