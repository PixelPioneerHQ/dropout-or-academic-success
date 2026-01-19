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

def test_prediction(base_url, data=None):
    """Test the prediction endpoint"""
    url = f"{base_url}/predict"
    logger.info(f"Testing prediction endpoint: {url}")
    
    # If no data provided, use example or create dummy data
    if data is None:
        example = test_example(base_url)
        if example:
            data = example
        else:
            logger.warning("Using dummy data for prediction")
            data = {
                "feature1": 0,
                "feature2": "value"
            }
    
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

def test_batch_prediction(base_url, num_samples=5, data=None):
    """Test the batch prediction endpoint"""
    url = f"{base_url}/predict/batch"
    logger.info(f"Testing batch prediction endpoint: {url}")
    
    # If no data provided, use example or create dummy data
    if data is None:
        example = test_example(base_url)
        if example:
            # Create multiple copies of the example
            data = [example] * num_samples
        else:
            logger.warning("Using dummy data for batch prediction")
            data = [{"feature1": i, "feature2": "value"} for i in range(num_samples)]
    
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
    parser.add_argument('--test', type=str, choices=['health', 'metadata', 'example', 'predict', 'batch', 'load', 'all'],
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
    
    if args.test == 'batch' or args.test == 'all':
        test_batch_prediction(args.url, 5, test_data)
    
    if args.test == 'load' or args.test == 'all':
        load_test(args.url, args.num_requests, args.concurrency)

if __name__ == "__main__":
    main()