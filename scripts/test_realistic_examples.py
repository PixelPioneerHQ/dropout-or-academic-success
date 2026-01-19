#!/usr/bin/env python
"""
Test the prediction API with realistic student data examples
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8080"

def load_test_data():
    """Load the test data examples"""
    with open('test_data_examples.json', 'r') as f:
        return json.load(f)

def test_single_prediction(student_name, student_data):
    """Test a single prediction"""
    print(f"\n🎓 Testing: {student_name}")
    print(f"Description: {student_data['description']}")
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=student_data['data'])
        response.raise_for_status()
        
        result = response.json()
        prediction = result['prediction']
        probabilities = result['probabilities']
        
        print(f"📊 Prediction: {prediction}")
        print(f"📈 Probabilities:")
        for outcome, prob in probabilities.items():
            print(f"  • {outcome}: {prob:.3f} ({prob*100:.1f}%)")
        
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_batch_prediction():
    """Test batch prediction with all examples"""
    print(f"\n📦 Testing Batch Prediction with all examples")
    
    test_data = load_test_data()
    batch_data = [student_data['data'] for student_data in test_data.values()]
    
    try:
        response = requests.post(f"{BASE_URL}/predict/batch", json=batch_data)
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Batch prediction successful for {result['count']} students")
        
        # Show results for each student
        student_names = list(test_data.keys())
        for i, (student_name, prediction_result) in enumerate(zip(student_names, result['results'])):
            print(f"  • {student_name}: {prediction_result['prediction']}")
        
        return result
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def main():
    """Main test function"""
    print("🚀 Testing Student Dropout Prediction API with Realistic Examples")
    print("="*60)
    
    # Check if API is running
    try:
        health_response = requests.get(f"{BASE_URL}/health", timeout=5)
        health_response.raise_for_status()
        print("✅ API is running and healthy")
    except Exception as e:
        print(f"❌ API is not accessible: {e}")
        print("Please make sure to run: python scripts/predict.py")
        return
    
    # Load test data
    test_data = load_test_data()
    print(f"📚 Loaded {len(test_data)} test scenarios")
    
    # Test each scenario individually
    results = {}
    for student_name, student_data in test_data.items():
        result = test_single_prediction(student_name, student_data)
        if result:
            results[student_name] = result
    
    # Test batch prediction
    batch_result = test_batch_prediction()
    
    # Summary
    print(f"\n📋 Summary")
    print("="*60)
    if results:
        predictions_summary = {}
        for student_name, result in results.items():
            prediction = result['prediction']
            predictions_summary[prediction] = predictions_summary.get(prediction, 0) + 1
        
        print("Individual predictions:")
        for outcome, count in predictions_summary.items():
            print(f"  • {outcome}: {count} student(s)")
    
    print("\n🎯 Expected Predictions (based on student profiles):")
    print("  • high_performing_student: Graduate (high grades, scholarship)")
    print("  • at_risk_student: Dropout (poor grades, financial issues)")  
    print("  • average_student: Enrolled (moderate performance)")
    print("  • older_student: Graduate (mature, motivated)")
    print("  • international_student: Graduate (high grades, scholarship)")

if __name__ == "__main__":
    main()