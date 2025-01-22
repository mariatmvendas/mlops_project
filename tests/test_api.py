"""
export MYENDPOINT=http://localhost:8000
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
