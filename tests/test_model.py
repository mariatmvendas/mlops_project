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
