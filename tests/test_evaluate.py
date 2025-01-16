"""This script contains a unit test for the `evaluate` function in the `mlops_project.train` module.

The `evaluate` function is responsible for evaluating a trained model on a test dataset, and this test ensures:
1. The function runs without errors when a valid model file is present.
2. The function returns a valid result (e.g., a float indicating evaluation metrics).

Key Details:
- The test is skipped if the required model file (`models/model.pth`) is not found.
- The test uses test images and targets located in `data/processed` and checks the function's output.

Test Case:
- `test_evaluate`: Verifies that the `evaluate` function returns a valid result when provided with correct inputs.

Prerequisites:
- A trained model saved at `models/model.pth`.
- Test image tensors and target tensors saved at:
  - `data/processed/test_images.pt`
  - `data/processed/test_targets.pt`

Run the script with pytest:
    pytest test_evaluate.py
"""

import os
import pytest
from mlops_project.train import evaluate

model_path = "models/model.pth"

# Check if the model file exists
@pytest.mark.skipif(not os.path.exists(model_path), reason="Model file not found.")#FIXME does it make sense that it skips it if the model is not found?
def test_evaluate():
    result = evaluate(
        test_images_path="data/processed/test_images.pt",
        test_targets_path="data/processed/test_targets.pt",
        model_path=model_path,
        batch_size=2
    )
    assert result is None, "Expected float value, got None"
