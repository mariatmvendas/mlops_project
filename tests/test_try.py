import os
from PIL import Image
import pytest
import torch
from mlops_project.data import check_images_size, organize_and_rename_images, convert_images_to_tensors

# Mock paths for testing
RAW_DATA_PATH = "fake/data/raw"
PREPROCESSED_DATA_PATH = "fake/data/processed"
LABELS = ["cloudy", "desert", "forest", "water"]


def setup():
    """
    Sets up the mock data environment for testing purposes.
    """
    print("Setting up mock data...")
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
    """
    print("Cleaning up mock data...")
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
    Test the check_images_size function to ensure it identifies if all images have the same size.
    """
    setup()
    try:
        result, target_size = check_images_size(RAW_DATA_PATH)
        assert result is True
        assert target_size == (100, 100)  # Since all images are of size 100x100
    finally:
        teardown()


def test_organize_and_rename_images():
    """
    Test the organize_and_rename_images function to ensure images are correctly organized into train and test folders.
    """
    setup()
    try:
        organize_and_rename_images(PREPROCESSED_DATA_PATH, RAW_DATA_PATH)

        # Check if train and test folders are created
        assert os.path.exists(os.path.join(PREPROCESSED_DATA_PATH, "train"))
        assert os.path.exists(os.path.join(PREPROCESSED_DATA_PATH, "test"))

        train_folder = os.path.join(PREPROCESSED_DATA_PATH, "train")
        test_folder = os.path.join(PREPROCESSED_DATA_PATH, "test")

        assert len(os.listdir(train_folder)) == 16  # 80% of 20 images
        assert len(os.listdir(test_folder)) == 4   # 20% of 20 images
    finally:
        teardown()


def test_convert_images_to_tensors():
    """
    Test the convert_images_to_tensors function to ensure images are correctly converted to tensors and saved.
    """
    setup()
    organize_and_rename_images(PREPROCESSED_DATA_PATH, RAW_DATA_PATH)
    try:
        convert_images_to_tensors('fake/data/processed/train', 'fake/data','train_images.pt', 'train_targets.pt')

        # Check if tensors are saved
        assert os.path.exists(os.path.join('fake/data', 'train_images.pt'))
        assert os.path.exists(os.path.join('fake/data', 'train_targets.pt'))

        # Load the tensors to check their shape
        image_tensor = torch.load(os.path.join('fake/data', 'train_images.pt'))
        target_tensor = torch.load(os.path.join('fake/data', 'train_targets.pt'))

        # Check that the tensors have the expected shape
        print(image_tensor.shape[0])
        assert image_tensor.shape[0] == 16  # 16 images in train folder
        assert target_tensor.shape[0] == 16  # 16 labels
    finally:
        teardown()


if __name__ == "__main__":
    pytest.main()
