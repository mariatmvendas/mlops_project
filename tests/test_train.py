"""
This script contains a unit test for the `train` function in the `mlops_project.train` module.

Key Functionalities Tested:
1. **Training Process**:
    - Verifies that the `train` function runs successfully with mock training data.
    - Ensures the training pipeline executes for a single epoch with a small batch size (2).
2. **Model Saving**:
    - Checks that the trained model is saved at the expected path (`models/model.pth`).

Assertions:
- Ensures the file `models/model.pth` exists after the training process is completed.

Usage:
- This test can be used to confirm the integration of the training process and the model saving functionality.
- Run the test using pytest:
    pytest test_train.py
"""

import os
from mlops_project.train import train

def test_train():
    train(
        config="config.yaml",
        train_images_path = "data/processed/train_images.pt",
        train_targets_path = "data/processed/train_targets.pt",
        batch_size = 2,
        num_epochs = 1,
        learning_rate = 0.001)
    assert os.path.exists("models/model.pth"), "Expected saved model path 'models/model.pth'"
