"""
Locust Performance Testing Script for FastAPI Application

This script uses Locust, a load testing tool, to simulate user interactions with a FastAPI application.
It defines tasks to test two endpoints of the application:
1. `GET /`: Simulates users visiting the root endpoint to retrieve API instructions.
2. `POST /inference_satellite/`: Simulates users uploading an image to get a predicted label from the satellite image classification model.

Key Features:
- The script simulates concurrent users making requests to both endpoints.
- For the `/inference_satellite/` endpoint, it sends a test image (`desert.jpg`) as part of the request.
- It logs failed requests for debugging purposes.

How to Use:
1. locust -f locustfile.py --host http://127.0.0.1:8000
2. open interface at http://127.0.0.1:8089

Requirements:
1. A running FastAPI application at http://127.0.0.1:8000.
2. A test image located at ./tests/desert.jpg. 

"""

from locust import HttpUser, between, task

class MyUser(HttpUser):
    """A simple Locust user class that defines the tasks to be performed by the users."""

    # Define the base URL of the application
    host = "http://127.0.0.1:8000"
    wait_time = between(1, 2)

    @task
    def get_root(self) -> None:
        """A task that simulates a user visiting the root URL of the FastAPI app."""
        self.client.get("/")

    @task
    def post_inference(self) -> None:
        """A task that simulates a POST request to the /inference_satellite/ endpoint."""
        with open("./tests/desert.jpg", "rb") as image_file:
            # Send the POST request with the image file
            response = self.client.post(
                "/inference_satellite/",
                files={"data": image_file},
            )
            if response.status_code != 200:
                print(f"Failed request: {response.status_code}, {response.text}")
