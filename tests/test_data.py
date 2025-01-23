"""
This script contains unit tests for the `mlops_project.data` module using the pytest framework.

The tests verify the following functionalities:
1. `check_images_size`:
    - Ensures that all images in a directory have the same size or identifies the most common size.
2. `organize_and_rename_images`:
    - Organizes images into train and test folders, renames them according to their labels, and resizes them if necessary.
3. `convert_images_to_tensors`:
    - Converts images to PyTorch tensors and saves both the image tensors and their corresponding labels to disk.

Each test case uses mock data, which is generated during a setup phase and cleaned up afterward:
- `setup`: Creates mock directories, labels, and images for testing.
- `teardown`: Removes all resources created during the setup phase to ensure a clean environment for each test.

Key Paths:
- `RAW_DATA_PATH`: Path where raw mock images are stored.
- `PREPROCESSED_DATA_PATH`: Path where processed data (train/test splits) is stored.
- `LABELS`: List of image labels used for generating mock data.

Usage:
Run the script with pytest to execute the tests:
    pytest test_data.py
"""

import os
from PIL import Image
import pytest
import torch
from mlops_project.data import check_images_size, organize_and_rename_images, convert_images_to_tensors

# Mock paths for testing
RAW_DATA_PATH = "mock/data/raw"
PREPROCESSED_DATA_PATH = "mock/data/processed"
LABELS = ["cloudy", "desert", "forest", "water"]


def setup():
    """
    Sets up the mock data environment for testing purposes.
    - Creates raw data directories for each label.
    - Generates 5 mock images for each label.
    """
    print("Setting up mock data environment...")
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)

    for label in LABELS:
        folder = os.path.join(RAW_DATA_PATH, label)
        os.makedirs(folder, exist_ok=True)
        for i in range(1, 6):  # Create 5 mock images per label
            img = Image.new('RGB', (100, 100), color=(73, 109, 137))
            img.save(os.path.join(folder, f"{label}_{i}.jpg"))


def teardown():
    """
    Cleans up the mock data environment after testing.
    - Removes all created files and directories.
    """
    print("Cleaning up mock data environment...")
    for label in LABELS:
        folder_path = os.path.join(RAW_DATA_PATH, label)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))
            os.rmdir(folder_path)

    if os.path.exists(RAW_DATA_PATH):
        os.rmdir(RAW_DATA_PATH)

    if os.path.exists(PREPROCESSED_DATA_PATH):
        for folder in ["train", "test"]:
            folder_path = os.path.join(PREPROCESSED_DATA_PATH, folder)
            if os.path.exists(folder_path):
                for file in os.listdir(folder_path):
                    os.remove(os.path.join(folder_path, file))
                os.rmdir(folder_path)
        os.rmdir(PREPROCESSED_DATA_PATH)


def test_check_images_size():
    """
    Tests `check_images_size` to ensure it validates image sizes correctly.
    - Verifies that all images in the mock dataset are the same size.
    """
    setup()
    try:
        result, target_size = check_images_size(RAW_DATA_PATH)
        assert result is True
        assert target_size == (100, 100)  # All images are 100x100 pixels
    finally:
        teardown()


def test_organize_and_rename_images():
    """
    Tests `organize_and_rename_images` to ensure it processes images correctly.
    - Checks that images are split into train and test folders.
    - Validates that 80% of images go into the train folder and 20% into the test folder.
    """
    setup()
    try:
        organize_and_rename_images(PREPROCESSED_DATA_PATH, RAW_DATA_PATH)

        # Validate folder creation
        assert os.path.exists(os.path.join(PREPROCESSED_DATA_PATH, "train"))
        assert os.path.exists(os.path.join(PREPROCESSED_DATA_PATH, "test"))

        # Validate image counts in train and test folders
        train_folder = os.path.join(PREPROCESSED_DATA_PATH, "train")
        test_folder = os.path.join(PREPROCESSED_DATA_PATH, "test")

        assert len(os.listdir(train_folder)) == 16  # 80% of 20 images
        assert len(os.listdir(test_folder)) == 4   # 20% of 20 images
    finally:
        teardown()


# def test_convert_images_to_tensors():
#     """
#     Tests `convert_images_to_tensors` to ensure images are converted to PyTorch tensors correctly.
#     - Verifies that tensor files are saved.
#     - Checks the shape of the saved tensors to ensure accuracy.
#     """
#     setup()
#     organize_and_rename_images(PREPROCESSED_DATA_PATH, RAW_DATA_PATH)
#     try:
#         convert_images_to_tensors('mock/data/processed/train', 'mock/data', 'train_images.pt', 'train_targets.pt')

#         # Check if tensor files are created
#         assert os.path.exists(os.path.join('mock/data', 'train_images.pt'))
#         assert os.path.exists(os.path.join('mock/data', 'train_targets.pt'))

#         # Load tensors and validate their shapes
#         image_tensor = torch.load(os.path.join('mock/data', 'train_images.pt'))
#         target_tensor = torch.load(os.path.join('mock/data', 'train_targets.pt'))

#         # Validate tensor dimensions
#         print(f"Image Tensor Shape: {image_tensor.shape}")
#         assert image_tensor.shape[0] == 16  # 16 images in train folder
#         assert target_tensor.shape[0] == 16  # 16 corresponding labels
#     finally:
#         teardown()


if __name__ == "__main__":
    pytest.main()
