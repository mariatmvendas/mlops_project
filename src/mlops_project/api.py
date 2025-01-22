""" Satellite Inference API

This script sets up a FastAPI application to provide a web interface for satellite image classification.

Key Features:
- **Root Endpoint (`GET /`)**:
  - Provides a welcome message and instructions on how to use the API.
  - Explains how to upload an image and get its classification.

- **Inference Endpoint (`POST /inference_satellite/`)**:
  - Accepts an image file as input via multipart form-data.
  - Converts the image to RGB format if necessary and saves it temporarily on the server.
  - Invokes an external script (`inference.py`) to perform satellite image classification using a pre-trained model.
  - Cleans up temporary files after processing, ensuring efficient resource usage.
  - Returns the predicted class label as a JSON response.

How It Works:
1. **Image Upload**:
   - The user uploads an image via the `/inference_satellite/` endpoint.
   - The uploaded image is converted to RGB format if needed and saved temporarily.

2. **Script Execution**:
   - The external script (`inference.py`) is executed using Python's `subprocess.run`.
   - The temporary image file path is passed as an argument to the script.

3. **Error Handling**:
   - Errors during script execution or processing are captured and returned in the API response.
   - Temporary image files are always deleted after processing, ensuring no leftover files remain.

4. **Platform Independence**:
   - Uses the `tempfile` module to create and manage temporary files in a cross-platform manner.
   - Avoids hardcoded environment variables, making the script adaptable to various systems without additional configuration.

Usage Example:
1. Start the server:
   ```bash
   uvicorn src.mlops_project.api:app --reload
   ```

2. Test the inference endpoint:
   ```bash
   curl -X POST "http://127.0.0.1:8000/inference_satellite/" \
        -H "Content-Type: multipart/form-data" \
        -F "data=@path_to_your_image.jpg"
   ```
"""

import subprocess
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import os
import tempfile

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
    Run satellite model inference using an external script. This version is designed to work on all systems.
    """
    # Use tempfile to create a temporary image file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
        temp_image_path = temp_image.name
        i_image = Image.open(data.file)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")
        i_image.save(temp_image_path)

    try:
        # Prepare the command and environment
        command = ["python", "src/mlops_project/inference.py", temp_image_path]

        # Run the subprocess without manually setting environment variables
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Check for errors in the subprocess
        if result.returncode != 0:
            return {"error": f"Inference script failed: {result.stderr}"}

        # Parse and return the result from the inference script
        predicted_class = result.stdout.strip()
        return {"predicted_class": predicted_class}
    except FileNotFoundError:
        return {"error": "The inference script could not be found. Ensure 'src/mlops_project/inference.py' exists."}
    except Exception as e:
        return {"error": f"An unexpected error occurred: {str(e)}"}
    finally:
        # Clean up the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
