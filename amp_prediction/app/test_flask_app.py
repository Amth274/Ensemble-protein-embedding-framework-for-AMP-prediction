#!/usr/bin/env python3
"""
Quick test script for the Flask AMP Prediction app
"""

import requests
import json
import sys
from pathlib import Path

def test_flask_app():
    """Test the Flask app endpoints."""
    base_url = "http://127.0.0.1:5000"

    print("Testing Flask AMP Prediction App")
    print("=" * 40)

    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/api/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ Health check:", health_data)
        else:
            print("❌ Health check failed:", response.status_code)
            return False
    except requests.exceptions.RequestException as e:
        print("❌ Could not connect to Flask app:", e)
        print("Make sure the Flask app is running with: python run_flask_app.py")
        return False

    # Test prediction endpoint
    test_sequence = "GIGKFLHSAKKFGKAFVGEIMNS"  # Magainin-2

    try:
        response = requests.post(
            f"{base_url}/api/predict",
            json={"sequence": test_sequence},
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction test successful:")
            print(f"  Sequence: {test_sequence}")
            print(f"  Prediction: {'AMP' if result['ensemble']['prediction'] == 1 else 'Non-AMP'}")
            print(f"  Confidence: {result['ensemble']['confidence']:.2%}")
            print(f"  Individual models: {len(result['individual'])} models")
        else:
            print("❌ Prediction test failed:", response.status_code)
            print("Response:", response.text)
            return False

    except requests.exceptions.RequestException as e:
        print("❌ Prediction test failed:", e)
        return False

    # Test batch endpoint with sample data
    test_sequences = [
        "GIGKFLHSAKKFGKAFVGEIMNS",  # Magainin-2
        "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQ"  # Non-AMP
    ]

    try:
        response = requests.post(
            f"{base_url}/api/batch",
            json={"sequences": test_sequences},
            timeout=15
        )

        if response.status_code == 200:
            results = response.json()
            print("✅ Batch prediction test successful:")
            print(f"  Processed {len(results['results'])} sequences")
            for i, result in enumerate(results['results']):
                pred = 'AMP' if result['ensemble']['prediction'] == 1 else 'Non-AMP'
                conf = result['ensemble']['confidence']
                print(f"  Sequence {i+1}: {pred} ({conf:.2%})")
        else:
            print("❌ Batch prediction test failed:", response.status_code)
            print("Response:", response.text)
            return False

    except requests.exceptions.RequestException as e:
        print("❌ Batch prediction test failed:", e)
        return False

    print("\n🎉 All tests passed! Flask app is working correctly.")
    return True

if __name__ == '__main__':
    success = test_flask_app()
    sys.exit(0 if success else 1)