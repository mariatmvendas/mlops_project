"""
This script
 1) trains a model to classify the train images in data/processed and saves a model.pth file to the models folder
 2) evaluates the model saved on a given path on the test images of data/processed by calculating accuracy


Hyperparameters:
These must be defined in a .yaml file in the configs folder with the desired values before running any code
 1) train function: train_images_path, train_targets_path, batch_size, num_epochs, learning_rate
 2) evaluate function: test_images_path, test_targets_path, model_path, batch_size

 The default .yaml file is config.yaml

Usage:
Call the functions in the script with Typer
This will run with the hyperparameters from the default .yaml file (config.yaml):
- python src/mlops_project/train.py train
- python src/mlops_project/train.py evaluate

To switch to a different .yaml file run:
- python src/mlops_project/train.py train --config exp1
- python src/mlops_project/train.py evaluate --config exp1

The .yaml file options are overriden by command line options.
This will run with the batch_size=3 while the other hyperparameters will follow the default .yaml file (config.yaml):
- python src/mlops_project/train.py evaluate --batch-size 3

"""

import typer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
import timm
import wandb
import os
from dotenv import load_dotenv
from hydra import initialize, compose


app = typer.Typer()

# Create logger
logger.remove()
logger.add("logs/log_debug.log", level="DEBUG", rotation="100 KB")

# Wandb configuration
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
if wandb_api_key:
    wandb.login(key=wandb_api_key)
else:
    typer.echo("Warning: WANDB_API_KEY is not set. Wandb logging will be disabled.")
    wandb.init(mode="disabled")

# Check device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.debug(f"Used device: {device}")


def train_dataloader_satellite():
    train_images_path: str = "data/processed/train_images.pt"
    train_targets_path: str = "data/processed/train_targets.pt"
    train_images = torch.load(train_images_path, weights_only=True)
    train_targets = torch.load(train_targets_path, weights_only=True)
    return train_images, train_targets


@app.command()
def train(
    config: str = typer.Option("config", help="Path to the config file", show_default=True),
    train_images_path: str = typer.Option(None, help="Path to the train images"),
    train_targets_path: str = typer.Option(None, help="Path to the train targets"),
    batch_size: int = typer.Option(None, help="Batch size for training"),
    learning_rate: float = typer.Option(None, help="Learning rate for optimizer"),
    num_epochs: int = typer.Option(None, help="Number of epochs for training")):


    """
    Train the model.

        Args:
            train_images_path (str): Path to the file of the train image tensors. Defaults to "data/processed/train_images.pt"
            train_targets_path (str): Path to the file of the train target tensors. Defaults to "data/processed/train_targets.pt"
            batch_size (int): Batch size. Defaults to 8
            num_epochs (int): Number of epochs. Defaults to 5
            learning_rate (float): Learning rate. Defaults to 0.001

        Returns:
            None

    """
    # Initialize wandb
    run = wandb.init(
        project="mlops_project",
        job_type="train",
        config={"learning_rate": learning_rate, "batch_size": batch_size, "num_epochs": num_epochs})

    with initialize(config_path="../../configs",job_name="test_app"):
        cfg = compose(config_name=config)

    # Override values from config with command-line options if provided
    train_images_path = train_images_path or cfg.train["train_images_path"]
    train_targets_path = train_targets_path or cfg.train["train_targets_path"]
    batch_size = batch_size or cfg.train["batch_size"]
    learning_rate = learning_rate or cfg.train["learning_rate"]
    num_epochs = num_epochs or cfg.train["num_epochs"]

    typer.echo(f"Configuration used for training:")
    typer.echo(f"  Train Images Path: {train_images_path}")
    typer.echo(f"  Train Targets Path: {train_targets_path}")
    typer.echo(f"  Batch Size: {batch_size}")
    typer.echo(f"  Learning Rate: {learning_rate}")
    typer.echo(f"  Number of Epochs: {num_epochs}")


    # Load training data
    try:
        train_images = torch.load(train_images_path, weights_only=True)
        train_targets = torch.load(train_targets_path, weights_only=True)
    except Exception as e:
        typer.echo(f"Error loading training data: {e}")
        raise typer.Exit()

    # Check dataset size
    typer.echo(f"train_images size: {train_images.size()}")
    typer.echo(f"train_targets size: {train_targets.size()}")
    logger.debug(f"train_images size: {train_images.size()}")
    logger.debug(f"train_targets size: {train_targets.size()}")

    # Ensure images are in CHW format if needed
    if train_images.shape[-1] == 3:  # HWC format
        train_images = torch.stack([img.permute(2, 0, 1) for img in train_images])

    # Use a subset of data for debugging (optional)
    subset_size = 20 # Adjust this as needed for debugging
    train_images = train_images[:subset_size]
    train_targets = train_targets[:subset_size]

    # Create Dataset and DataLoader
    dataset = TensorDataset(train_images, train_targets)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the model
    model = timm.create_model("resnet18", pretrained=True, num_classes=4)
    model = model.to(device)
    logger.debug(f"Model: {model}")

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            wandb.log({"train_loss": loss.item()})

        typer.echo(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "models/model.pth")
    typer.echo("Training complete! Model saved as model.pth and run logged on W&B.")

@app.command()
def evaluate(
    config: str = typer.Option("config", help="Path to the config file", show_default=True),
    test_images_path: str = typer.Option(None, help="Path to the test images"),
    test_targets_path: str = typer.Option(None, help="Path to the test targets"),
    batch_size: int = typer.Option(None, help="Batch size for evaluating"),
    model_path: int = typer.Option(None, help="Model path for evaluating")):

    """
    Evaluate the model.

        Args:
            test_images_path (str): Path to the file of the test image tensors. Defaults to "data/processed/test_images.pt"
            test_targets_path (str): Path to the file of the test target tensors. Defaults to "data/processed/test_targets.pt"
            model_path (str): Name of the file containing the saved model. Defaults to "model.pth"
            batch_size (int): Batch size. Defaults to 8

        Returns:
            None

    """
    # Initialize wandb
    run = wandb.init(
        project="mlops_project",
        job_type="evaluate",
        config={"batch_size": batch_size})

    with initialize(config_path="../../configs",job_name="test_app"):
        cfg = compose(config_name=config)

    # Override values from config with command-line options if provided
    test_images_path = test_images_path or cfg.evaluate["test_images_path"]
    test_targets_path = test_targets_path or cfg.evaluate["test_targets_path"]
    batch_size = batch_size or cfg.evaluate["batch_size"]
    model_path = model_path or cfg.evaluate["model_path"]

    typer.echo(f"Configuration used for evaluating:")
    typer.echo(f"  Test Images Path: {test_images_path}")
    typer.echo(f"  Test Targets Path: {test_targets_path}")
    typer.echo(f"  Batch Size: {batch_size}")
    typer.echo(f"  Model Path: {model_path}")


    # Load test data
    try:
        test_images = torch.load(test_images_path, weights_only=True)
        test_targets = torch.load(test_targets_path, weights_only=True)
    except Exception as e:
        typer.echo(f"Error loading test data: {e}")
        raise typer.Exit()

    # Check dataset size
    typer.echo(f"test_images size: {test_images.size()}")
    typer.echo(f"test_targets size: {test_targets.size()}")
    logger.debug(f"test_images size: {test_images.size()}")
    logger.debug(f"test_targets size: {test_targets.size()}")

    # Ensure images are in CHW format if needed
    if test_images.shape[-1] == 3:  # HWC format
        test_images = torch.stack([img.permute(2, 0, 1) for img in test_images])

    # Create Dataset and DataLoader
    test_dataset = TensorDataset(test_images, test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    model = timm.create_model("resnet18", pretrained=True, num_classes=4)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model = model.to(device)

    # Evaluate the model
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_dataloader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    typer.echo(f"Accuracy on test set: {accuracy:.2f}%")
    logger.debug(f"Accuracy on test set: {accuracy:.2f}%")
    wandb.log({"validation_accuracy": accuracy})

    artifact = wandb.Artifact(
        name="mlops_project",
        type="model",
        description="A model trained to classify satellite images",
        metadata={"validation_accuracy": accuracy},
    )
    artifact.add_file("models/model.pth")
    run.log_artifact(artifact)

if __name__ == "__main__":
    app()



# # PROFILING
# with profile(activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
#     for i in range(10):
#         model(images)
#         prof.step()

# # Print profiling results sorted by self CPU time
# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))


# # Print profiling results sorted by self CPU memory usage
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
# print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=30))

# prof.export_chrome_trace("trace.json")

""""
This script use the model and the data module to start the training.

"""

from pytorch_lightning import Trainer
from mlops_project.data import SatelliteDataModule
from mlops_project.model import SatelliteModel

def train_model():
    data_module = SatelliteDataModule(
        train_csv="./data/raw/train_labels.csv",
        test_csv="./data/raw/test_labels.csv",
        img_dir="./data/raw/train_images",
        batch_size=32
    )
    model = SatelliteModel(learning_rate=0.001)
    trainer = Trainer(max_epochs=10, gpus=1)
    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    train_model()
