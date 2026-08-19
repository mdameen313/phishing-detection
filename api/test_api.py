# api/test_api.py
"""
Test the API endpoints
"""

import requests
import json

# API URL
BASE_URL = "http://127.0.0.1:5000"

def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print(f"Health: {response.json()}")

def test_predict():
    """Test prediction endpoint"""
    test_urls = [
        "https://www.google.com",
        "https://hdfc-secure-login.xyz",
        "http://bit.ly/3xYZ123",
    ]
    
    for url in test_urls:
        response = requests.post(
            f"{BASE_URL}/predict",
            json={"url": url}
        )
        result = response.json()
        print(f"\nURL: {url}")
        print(f"  Verdict: {result.get('verdict', 'ERROR')}")
        print(f"  Confidence: {result.get('confidence', 0):.1%}")
        print(f"  Layer: {result.get('layer', 'Unknown')}")
        print(f"  Cached: {result.get('cached', False)}")

def test_batch():
    """Test batch prediction"""
    urls = [
        "https://www.amazon.com",
        "https://secure-login.paypal-verify.xyz",
    ]
    
    response = requests.post(
        f"{BASE_URL}/predict_batch",
        json={"urls": urls}
    )
    print(f"\nBatch Results:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("="*60)
    print("TESTING API")
    print("="*60)
    
    print("\n1. Testing Health...")
    test_health()
    
    print("\n2. Testing Predictions...")
    test_predict()
    
    print("\n3. Testing Batch...")
    test_batch()
    
    print("\n✅ API Test Complete!")