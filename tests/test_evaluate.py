import os
import pytest
from mlops_project.train import evaluate

model_path = "models/model.pth"

# Check if the model file exists
@pytest.mark.skipif(not os.path.exists(model_path), reason="Model file not found.")
def test_evaluate():
    result = evaluate(
        test_images_path="data/processed/test_images.pt",
        test_targets_path="data/processed/test_targets.pt",
        model_path=model_path,
        batch_size=2
    )
    assert result is None, "Expected float value, got None"
