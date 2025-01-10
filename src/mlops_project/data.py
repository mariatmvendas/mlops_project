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

def check_images_size(directory):
    """Check if all images in the directory have the same size and return the target size."""
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
        # Choose the most common image size as the target size
        target_size = max(image_sizes, key=lambda x: list(image_sizes).count(x))
        print(f"Resizing images to: {target_size}")
        return False, target_size


def organize_and_rename_images():
    """Organize images by their labels, rename them (cloudy_1, cloudy_2, ...), resize them if necessary, and split into train/test."""
    
    # Create train and test folders
    train_folder = os.path.join(PREPROCESSED_DATA_PATH, "train")
    test_folder = os.path.join(PREPROCESSED_DATA_PATH, "test")
    
    if not os.path.exists(train_folder):
        os.makedirs(train_folder)
        print(f"Created folder: {train_folder}")
    
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"Created folder: {test_folder}")
    
    # Check image sizes in the raw data before organizing and renaming
    all_same_size, target_size = check_images_size(RAW_DATA_PATH)
    
    if not all_same_size:
        print("Resizing images to the target size...")
    
    for folder_name in LABELS:
        source_folder = os.path.join(RAW_DATA_PATH, folder_name)
        
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


def convert_images_to_tensors(input_folder, image_output_file, target_output_file):
    """Convert images to tensors and save as .pt"""
    images = []
    targets = []
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust image extensions as needed
            # Extract the label from the filename (assuming the label is part of the filename)
            label = filename.split('_')[0]  # Adjust if needed based on your filename format
            if label in LABELS:
                label_index = LABELS.index(label)  # Get the index of the label in the LABELS list
                print(label_index)
                # Load the image and apply transformation
                img_path = os.path.join(input_folder, filename)
                img = Image.open(img_path).convert("RGB")  # Open image and ensure it has 3 channels
                tensor = transform(img)
                
                # Append the image tensor and label index to the respective lists
                images.append(tensor)
                targets.append(label_index)
    
    # Stack the images into a single tensor (batch of images)
    image_tensor = torch.stack(images)
    target_tensor = torch.tensor(targets)  # Convert the list of labels to a tensor
    
    # Save the tensors to .pt files
    torch.save(image_tensor, os.path.join(PREPROCESSED_DATA_PATH, image_output_file))
    torch.save(target_tensor, os.path.join(PREPROCESSED_DATA_PATH, target_output_file))
    print(f"Saved image tensor to {os.path.join(PREPROCESSED_DATA_PATH, image_output_file)} and target tensor to {os.path.join(PREPROCESSED_DATA_PATH, target_output_file)}")

# Organize images and convert them to tensors
organize_and_rename_images()

# Convert and save images and targets from train and test folders
convert_images_to_tensors('data/processed/train', 'train_images.pt', 'train_targets.pt')
convert_images_to_tensors('data/processed/test', 'test_images.pt', 'test_targets.pt')