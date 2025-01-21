
import numpy as np  # Import numpy first to avoid threading conflicts
import argparse
import torch
import timm
from torchvision import transforms
from PIL import Image
import argparse
import torch
import timm
from torchvision import transforms
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Labels for the classification
LABELS = ["cloudy", "desert", "forest", "water"]

# Define a function to load the model
def load_model(model_path: str = "models/model.pth"):
    """
    Load the pre-trained model with weights from the specified path.
    Args:
        model_path (str): Path to the saved model. Defaults to "models/model.pth".
    Returns:
        torch.nn.Module: The loaded model.
    """
    model = timm.create_model("resnet18", pretrained=True, num_classes=len(LABELS))  # Adjust `num_classes` as per your model
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

# Define a function for preprocessing an image
def preprocess_image(image_path: str):
    """
    Preprocess the input image for inference.
    Args:
        image_path (str): Path to the image file.
    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to the input size of ResNet
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Define a function to make a prediction
def predict(model, image_tensor):
    """
    Perform inference on a single image.
    Args:
        model (torch.nn.Module): The trained model.
        image_tensor (torch.Tensor): The preprocessed image tensor.
    Returns:
        str: The predicted class label.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    return LABELS[predicted_class.item()]  # Map the predicted class index to the label

# Main function
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Image Classification Inference")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    args = parser.parse_args()
    
    # Fixed model path
    model_path = "models/model.pth"
    
    # Load the model
    model = load_model(model_path)
    
    # Preprocess the image
    image_tensor = preprocess_image(args.image_path)
    
    # Make a prediction
    predicted_label = predict(model, image_tensor)
    print(f"Predicted label: {predicted_label}")
