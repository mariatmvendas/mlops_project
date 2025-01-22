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
            # Optionally, print or log the response for debugging
            if response.status_code != 200:
                print(f"Failed request: {response.status_code}, {response.text}")
