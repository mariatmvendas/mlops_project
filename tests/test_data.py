import os
import pytest
from PIL import Image
import torch
import sys
import os

from mlops_project.data import check_images_size, organize_and_rename_images, convert_images_to_tensors


# Mock paths for testing
RAW_DATA_PATH = "data/raw"
PREPROCESSED_DATA_PATH = "data/processed"
LABELS = ["cloudy", "desert", "forest", "water"]

@pytest.fixture
def create_mock_data():
    """
    Creates mock data for testing purposes.
    """
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    os.makedirs(PREPROCESSED_DATA_PATH, exist_ok=True)

    # Creating mock images in each label folder
    for label in LABELS:
        os.makedirs(os.path.join(RAW_DATA_PATH, label), exist_ok=True)
        for i in range(5):  # 5 images per folder
            img = Image.new('RGB', (100, 100), color=(73, 109, 137))  # Create a 100x100 image
            img.save(os.path.join(RAW_DATA_PATH, label, f"{label}_{i}.jpg"))
    
    yield

    # Cleanup after tests
    for label in LABELS:
        folder_path = os.path.join(RAW_DATA_PATH, label)
        for file in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, file))
        os.rmdir(folder_path)
    os.rmdir(RAW_DATA_PATH)
    os.rmdir(PREPROCESSED_DATA_PATH)


# def test_check_images_size(create_mock_data):
#     """
#     Test the check_images_size function to ensure it identifies if all images have the same size.
#     """
#     result, target_size = check_images_size(RAW_DATA_PATH)
#     assert result is True
#     assert target_size == (100, 100)  # Since all images are of size 100x100


# def test_check_images_size_different_sizes(create_mock_data):
#     """
#     Test if the check_images_size function can handle different image sizes.
#     """
#     # Create a new folder with different size images
#     os.makedirs(os.path.join(RAW_DATA_PATH, "forest"), exist_ok=True)
#     img1 = Image.new('RGB', (100, 100), color=(255, 0, 0))
#     img1.save(os.path.join(RAW_DATA_PATH, "forest", "forest_1.jpg"))
#     img2 = Image.new('RGB', (200, 200), color=(0, 255, 0))
#     img2.save(os.path.join(RAW_DATA_PATH, "forest", "forest_2.jpg"))

#     result, target_size = check_images_size(RAW_DATA_PATH)
#     assert result is False
#     assert target_size == (100, 100)  # Should return the most common size


def test_organize_and_rename_images(create_mock_data):
    """
    Test the organize_and_rename_images function to ensure images are correctly organized into train and test folders.
    """
    organize_and_rename_images()

    # Check if train and test folders are created
    assert os.path.exists(os.path.join(PREPROCESSED_DATA_PATH, "train"))
    assert os.path.exists(os.path.join(PREPROCESSED_DATA_PATH, "test"))

    # Check if images are moved and renamed in the folders
    for label in LABELS:
        train_folder = os.path.join(PREPROCESSED_DATA_PATH, "train", label)
        test_folder = os.path.join(PREPROCESSED_DATA_PATH, "test", label)

        assert len(os.listdir(train_folder)) == 4  # 80% of 5 images
        assert len(os.listdir(test_folder)) == 1   # 20% of 5 images


def test_convert_images_to_tensors(create_mock_data):
    """
    Test the convert_images_to_tensors function to ensure images are correctly converted to tensors and saved.
    """
    convert_images_to_tensors('data/processed/train', 'train_images.pt', 'train_targets.pt')

    # Check if tensors are saved
    assert os.path.exists(os.path.join(PREPROCESSED_DATA_PATH, 'train_images.pt'))
    assert os.path.exists(os.path.join(PREPROCESSED_DATA_PATH, 'train_targets.pt'))

    # Load the tensors to check their shape
    image_tensor = torch.load(os.path.join(PREPROCESSED_DATA_PATH, 'train_images.pt'))
    target_tensor = torch.load(os.path.join(PREPROCESSED_DATA_PATH, 'train_targets.pt'))

    # Check that the tensors have the expected shape
    assert image_tensor.shape[0] == 4  # 4 images in train folder
    assert target_tensor.shape[0] == 4  # 4 labels


if __name__ == "__main__":
    pytest.main()
