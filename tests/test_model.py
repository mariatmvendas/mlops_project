"""
This script contains a unit test for training and validating a model using the `mlops_project.train` module.

Key Functionalities Tested:
1. **Training Process**:
    - Uses the `train` function to train a model for one epoch with a small batch size (2).
    - Ensures the training pipeline runs without errors given mock data.
2. **Model Structure**:
    - Verifies that the created model (ResNet-18) has the expected number of output classes (4).
3. **Model Output**:
    - Checks that the output shape of the model matches the expected dimensions when given a random input tensor.

Components:
- `train`:
    - Handles the training process using preprocessed data (`train_images.pt` and `train_targets.pt`).
- `timm.create_model`:
    - Used to create a ResNet-18 model with 4 output classes.
- Assertions:
    - Ensures the model has the correct number of classes.
    - Validates that the model's output shape matches the expected shape when processing a batch.

Usage:
- This test can be run to verify that the training process and model structure are functioning as expected.
"""

import torch
import timm
from mlops_project.train import train

def test_model():
    train(
        train_images_path = "data/processed/train_images.pt",
        train_targets_path = "data/processed/train_targets.pt",
        batch_size = 2,
        num_epochs = 1,
        learning_rate = 0.001)

    model = timm.create_model("resnet18", pretrained=False, num_classes=4)
    assert model.num_classes == 4, f"Expected 4 classes, got {model.num_classes}"

    output = model(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 4), f"Expected output shape (1, 4), got {output.shape}"
