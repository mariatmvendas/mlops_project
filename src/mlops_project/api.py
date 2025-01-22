"""
uvicorn --reload --port 8000 src.mlops_project.api:app
curl -X POST "http://127.0.0.1:8000/inference_satellite/" -H "Content-Type: multipart/form-data" -F "data=@./desert.jpg"
{"predicted_class":"Predicted label: desert"}(base)

Satellite Inference API

This script sets up a FastAPI application to provide a web interface for satellite image classification.

Key Features:
- **Root Endpoint (`GET /`)**:
  - Provides a welcome message and instructions on how to use the API.
  - Explains how to upload an image and get its classification.

- **Inference Endpoint (`POST /inference_satellite/`)**:
  - Accepts an image file as input via multipart form-data.
  - Saves the image temporarily to the server.
  - Calls an external script (`inference.py`) to perform satellite image classification using a pre-trained model.
  - Returns the predicted class label as a JSON response.

How It Works:
1. **Image Upload**:
   - The user uploads an image via the `/inference_satellite/` endpoint.
   - The image is saved temporarily in the server.

2. **Script Execution**:
   - The external script (`inference.py`) is invoked using Python's `subprocess.run`.
   - The temporary image file path is passed as an argument to the script.

3. **Environment Handling**:
   - To avoid threading conflicts with Intel MKL and OpenMP libraries, the `MKL_THREADING_LAYER` environment variable is set to `GNU`.

4. **Error Handling**:
   - If the script fails or returns an error, it is captured and returned in the API response.
   - The temporary image file is deleted after processing, regardless of success or failure.

Usage Example:
1. Start the server:
   ```bash
   uvicorn api:app --reload
"""


import subprocess
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import os

app = FastAPI()

@app.get("/")
async def root():
    """
    Root endpoint to explain how to use the API and provide inference capabilities.
    """
    return {
        "message": "Welcome to the Satellite Inference API!",
        "instructions": {
            "satellite_inference": "Use the POST /inference_satellite/ endpoint with an image file to classify the image.",
            "example_curl": "curl -X POST 'http://127.0.0.1:8000/inference_satellite/' -H 'Content-Type: multipart/form-data' -F 'data=@path_to_your_image.jpg'"
        }
    }


@app.post("/inference_satellite/")
async def inference_satellite(data: UploadFile = File(...)):
    """
    Run satellite model inference using an external script. This version is compatible with various systems.
    """
    # Save the uploaded image temporarily
    temp_image_path = "temp_image.jpg"
    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
    i_image.save(temp_image_path)

    try:
        # Prepare the command and environment
        command = ["python", "src/mlops_project/inference.py", temp_image_path]
        
        # Use the existing environment variables
        environment = os.environ.copy()
        
        # Optionally set MKL_THREADING_LAYER if it's compatible
        threading_layer = "GNU"  # You can change this to another threading backend if needed
        environment["MKL_THREADING_LAYER"] = threading_layer

        # Run the subprocess
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment  # Pass the environment
        )

        # Check for errors in the subprocess
        if result.returncode != 0:
            return {"error": f"Inference script failed: {result.stderr}"}

        # Parse and return the result from the inference script
        predicted_class = result.stdout.strip()
        return {"predicted_class": predicted_class}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
    finally:
        # Clean up the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
