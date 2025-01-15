import os
from mlops_project.train import train

def test_train():
    train(
        train_images_path = "data/processed/train_images.pt",
        train_targets_path = "data/processed/train_targets.pt",
        batch_size = 2,
        num_epochs = 1,
        learning_rate = 0.001)
    assert os.path.exists("models/model.pth"), "Expected saved model path 'models/model.pth'"
