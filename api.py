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
    """Run satellite model inference using an external script."""
    # Save the uploaded image temporarily
    temp_image_path = "temp_image.jpg"
    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")
    i_image.save(temp_image_path)

    try:
        # Call the inference script with MKL_THREADING_LAYER environment variable
        result = subprocess.run(
            ["python", "src/mlops_project/inference.py", temp_image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "MKL_THREADING_LAYER": "GNU"}  # Set threading layer to GNU
        )

        # Check for errors in the subprocess
        if result.returncode != 0:
            return {"error": f"Inference script failed: {result.stderr}"}

        # Parse and return the result from the inference script
        predicted_class = result.stdout.strip()
        return {"predicted_class": predicted_class}
    finally:
        # Clean up the temporary image file
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
