import subprocess
from contextlib import asynccontextmanager
import os
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor

# Lifespan function to load models
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up models on startup and shutdown."""
    global caption_model, feature_extractor, tokenizer, device, gen_kwargs
    print("Loading models...")

    # Load captioning model
    caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    caption_model.to(device)
    gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}

    yield

    print("Cleaning up models...")
    del caption_model, feature_extractor, tokenizer, device, gen_kwargs


# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    """Root endpoint to confirm the API is running."""
    return {"message": "Welcome to the Image Captioning and Satellite Inference API!"}


@app.post("/caption/")
async def caption(data: UploadFile = File(...)):
    """Generate a caption for an image."""
    i_image = Image.open(data.file)
    if i_image.mode != "RGB":
        i_image = i_image.convert(mode="RGB")

    pixel_values = feature_extractor(images=[i_image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = caption_model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return {"caption": preds[0].strip()}


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
        # Call the inference script
        result = subprocess.run(
            ["python", "src/mlops_project/inference.py", temp_image_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
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
