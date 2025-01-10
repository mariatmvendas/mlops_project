import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from PIL import Image
import os
import pandas as pd

# Dataset Class
class SatelliteImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Load data
def load_data(data_dir):
    image_paths, labels = [], []
    classes = sorted(os.listdir(data_dir))  # Assuming folder structure
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            image_paths.append(os.path.join(cls_dir, img_name))
            labels.append(class_to_idx[cls_name])
   
    return image_paths, labels, class_to_idx

# Model Definition
class SatelliteClassifier(nn.Module):
    def __init__(self, num_classes):
        super(SatelliteClassifier, self).__init__()
        self.model = timm.create_model('resnet50', pretrained=True)  # Use a pre-trained model
        in_features = self.model.get_classifier().in_features
        self.model.fc = nn.Linear(in_features, num_classes)  # Modify classifier for our task

    def forward(self, x):
        return self.model(x)

# Main Training and Evaluation
def train_and_evaluate(data_dir, batch_size=32, epochs=10, lr=0.001):
    # Prepare data
    image_paths, labels, class_to_idx = load_data(data_dir)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match model input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SatelliteImageDataset(train_paths, train_labels, transform=transform)
    val_dataset = SatelliteImageDataset(val_paths, val_labels, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model, Loss, Optimizer
    num_classes = len(class_to_idx)
    model = SatelliteClassifier(num_classes=num_classes)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training Loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
       
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=list(class_to_idx.keys())))

# Run the training
data_dir = "path_to_your_data"  # Replace with your data directory
train_and_evaluate(data_dir)