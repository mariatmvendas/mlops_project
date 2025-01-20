"""
This script preprocesses image datasets for machine learning workflows. 

It performs the following steps:
1. **Check Image Sizes**: 
    - Validates if all images in the `data/raw` folder are of the same size. 
    - If not, resizes them to the most common size.

2. **Organize and Rename Images**:
    - Splits images into train/test sets (80% train, 20% test).
    - Renames images by their labels (e.g., `cloudy_1.jpg`).
    - Stores the processed images in `data/processed/train` and `data/processed/test`.

3. **Convert Images to Tensors**:
    - Converts images into PyTorch tensors in batches.
    - Saves the tensors as `.pt` files (e.g., `train_images.pt` and `train_targets.pt`).

Paths:
- **RAW_DATA_PATH**: The path containing the raw images (`data/raw`).
- **PREPROCESSED_DATA_PATH**: The path where processed data is saved (`data/processed`).
- **LABELS**: A list of predefined image categories (`["cloudy", "desert", "forest", "water"]`).

Usage:
- Run the script directly to organize images and save them as tensors:
    `python src/mlops_project/data.py`

"""

import os
import random
import torch
from PIL import Image
from torchvision import transforms

# Define paths
RAW_DATA_PATH = "data/raw"
PREPROCESSED_DATA_PATH = "data/processed"
LABELS = ["cloudy", "desert", "forest", "water"]

# Define a transformation to convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert images to tensor
])

def check_images_size(directory: str) -> tuple[bool, tuple[int, int]]:
    """
    Check if all images in the directory have the same size and return the target size.

        Args:
            directory (str): Path to the directory of the raw data

        Returns:
            bool: True if all images have the same size, false otherwise
            tuple[int, int]: Target size of the image in width x height format, if the images have different sizes it chooses the most common
    """
    image_sizes = set()  # Use a set to track unique image sizes

    # Loop through all the files in the directory
    for folder_name in os.listdir(directory):
        folder_path = os.path.join(directory, folder_name)

        if os.path.isdir(folder_path):
            print(f"Checking images in folder: {folder_name}")

            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)

                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    with Image.open(file_path) as img:
                        image_size = img.size  # (width, height)
                        image_sizes.add(image_size)

    # Check if all images have the same size
    if len(image_sizes) == 1:
        print("All images have the same size.")
        return True, next(iter(image_sizes))  # Return the target size if they are all the same
    else:
        print(f"Images have different sizes: {image_sizes}")
        #FIXME DO WE REALLY NEED TO DO THIS?
        # Choose the most common image size as the target size
        target_size = max(image_sizes, key=lambda x: list(image_sizes).count(x))
        print(f"Resizing images to: {target_size}")
        return False, target_size


def organize_and_rename_images(preprocessed= PREPROCESSED_DATA_PATH, raw_data_path= RAW_DATA_PATH) -> None:
    """
    Organize images by their labels, rename them (cloudy_1, cloudy_2, ...), resize them if necessary, and split into train/test.

        Args:
            None

        Returns:
            None
    """
    # Create train and test folders
    train_folder = os.path.join(preprocessed, "train")
    test_folder =  os.path.join(preprocessed, "test")

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        print(f"Created folder: {train_folder}")

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"Created folder: {test_folder}")

    # Check image sizes in the raw data before organizing and renaming
    all_same_size, target_size = check_images_size(raw_data_path)

    if not all_same_size:
        print("Resizing images to the target size...")

    for folder_name in LABELS:
        source_folder = os.path.join(raw_data_path, folder_name)

        if not os.path.exists(source_folder):
            print(f"Source folder {source_folder} does not exist. Skipping.")
            continue

        # Get the list of images in the source folder
        images = [file_name for file_name in os.listdir(source_folder) if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Shuffle the images for random splitting
        random.shuffle(images)

        # Split images into train and test sets (80/20)
        split_index = int(0.8 * len(images))
        train_images = images[:split_index]
        test_images = images[split_index:]

        # Organize train images
        for counter, file_name in enumerate(train_images, start=1):
            source_file = os.path.join(source_folder, file_name)
            target_file_name = f"{folder_name}_{counter}{os.path.splitext(file_name)[1]}"
            target_file = os.path.join(train_folder, target_file_name)

            # Resize the image if necessary and move to train folder
            with Image.open(source_file) as img:
                if img.size != target_size:
                    img = img.resize(target_size)
                    print(f"Resized {file_name} to {target_size}")
                img.save(target_file)

            print(f"Moved and renamed {file_name} to {target_file_name}.")

        # Organize test images
        for counter, file_name in enumerate(test_images, start=1):
            source_file = os.path.join(source_folder, file_name)
            target_file_name = f"{folder_name}_{counter + len(train_images)}{os.path.splitext(file_name)[1]}"
            target_file = os.path.join(test_folder, target_file_name)

            # Resize the image if necessary and move to test folder
            with Image.open(source_file) as img:
                if img.size != target_size:
                    img = img.resize(target_size)
                    #print(f"Resized {file_name} to {target_size}")
                img.save(target_file)

            #print(f"Moved and renamed {file_name} to {target_file_name}.")


def convert_images_to_tensors(input_folder: str, output_folder: str, image_output_file: str, target_output_file: str, batch_size: int = 100) -> None:
    """
    Convert images to tensors in batches and save as .pt file.

        Args:
            input_folder (str): Path to the directory of the processed data
            image_output_file (str): Name of the output file of the image tensors
            target_output_file (str): Name of the output file of the target tensors
            batch_size (int): Number of images to process at once. Defaults to 100.

        Returns:
            None
    """
    images = []
    targets = []
    file_list = [f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png"))]

    for i, filename in enumerate(file_list):
        label = filename.split('_')[0]
        if label in LABELS:
            label_index = LABELS.index(label)
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img)

            images.append(tensor)
            targets.append(label_index)

            # Save batch to file
            if (i + 1) % batch_size == 0 or i == len(file_list) - 1:
                image_tensor = torch.stack(images)
                target_tensor = torch.tensor(targets)
                torch.save(image_tensor, os.path.join(output_folder, f"batch_{i // batch_size}_{image_output_file}"))
                torch.save(target_tensor, os.path.join(output_folder, f"batch_{i // batch_size}_{target_output_file}"))
                images.clear()
                targets.clear()

    print(f"Saved tensors in batches to {output_folder}")


if __name__ == "__main__":
    # Organize images and convert them to tensors
    organize_and_rename_images()

    # Convert and save images and targets from train and test folders
    convert_images_to_tensors('data/processed/train',PREPROCESSED_DATA_PATH, 'train_images.pt', 'train_targets.pt')
    convert_images_to_tensors('data/processed/test',PREPROCESSED_DATA_PATH, 'test_images.pt', 'test_targets.pt')


""""
This script is used to use PyTorch LightningDataModule to manage your dataset, loading images and labels from your CSV files.

"""

import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

class SatelliteDataModule(pl.LightningDataModule):
    def __init__(self, train_csv, test_csv, img_dir, batch_size=32):
        super().__init__()
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def setup(self, stage=None):
        self.train_dataset = SatelliteDataset(self.train_csv, self.img_dir, self.transform)
        self.test_dataset = SatelliteDataset(self.test_csv, self.img_dir, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
