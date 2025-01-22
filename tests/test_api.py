"""
Unit Test for the Root Endpoint of the FastAPI Application

This script tests the `/` (root) endpoint of the FastAPI application using TestClient.
The root endpoint is expected to:
1. Be accessible via a GET request.
2. Return a 200 OK status code.
3. Provide a JSON response containing a welcome message and instructions for using the API.

Key Features of the Test:
- Verifies that the response contains the correct HTTP status code.
- Checks that the JSON response includes:
  - A "message" key with a predefined welcome message.
  - An "instructions" key with relevant keys for API usage details.
- Ensures the structure and content of the API response align with expectations.

How to Use:
1.Run the API
2. Run the test using a testing framework like `pytest`:
   ```bash
   export MYENDPOINT=http://localhost:8000
   pytest path_to_test_file.py

"""
from fastapi.testclient import TestClient
from src.mlops_project.api import app

client = TestClient(app)

def test_read_root():
    """
    Test the root endpoint of the FastAPI application.
    """
    # Make a GET request to the root endpoint
    response = client.get("/")
    
    # Check the status code
    assert response.status_code == 200

    # Check the structure of the JSON response
    expected_message = "Welcome to the Satellite Inference API!"
    response_json = response.json()
    assert "message" in response_json
    assert response_json["message"] == expected_message

    # Optionally check instructions for additional robustness
    assert "instructions" in response_json
    assert "satellite_inference" in response_json["instructions"]
    assert "example_curl" in response_json["instructions"]
