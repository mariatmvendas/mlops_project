import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import timm

# Check device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load data
try:
    train_images = torch.load("data/processed/train_images.pt")
    train_targets = torch.load("data/processed/train_targets.pt")
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Check dataset size
print(f"train_images size: {train_images.size()}")
print(f"train_targets size: {train_targets.size()}")

# Ensure images are in CHW format if needed
if train_images.shape[-1] == 3:  # HWC format
    train_images = torch.stack([img.permute(2, 0, 1) for img in train_images])

# Use a subset of data for debugging (optional)
subset_size = 20  # Adjust this as needed for debugging
train_images = train_images[:subset_size]
train_targets = train_targets[:subset_size]

# Create Dataset and DataLoader
dataset = TensorDataset(train_images, train_targets)
batch_size = 8  # Smaller batch size to fit into memory
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the model
model = timm.create_model("resnet18", pretrained=True, num_classes=4)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for images, targets in dataloader:
        # Move data to the appropriate device
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    # Print epoch loss
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

print("Training complete!")

# Load the test data
try:
    test_images = torch.load("data/processed/test_images.pt")
    test_targets = torch.load("data/processed/test_targets.pt")
except Exception as e:
    print(f"Error loading test data: {e}")
    exit()

# Check test data sizes
print(f"test_images size: {test_images.size()}")
print(f"test_targets size: {test_targets.size()}")

# Ensure images are in CHW format if needed
if test_images.shape[-1] == 3:  # HWC format
    test_images = torch.stack([img.permute(2, 0, 1) for img in test_images])

# Create test Dataset and DataLoader
test_dataset = TensorDataset(test_images, test_targets)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Move model to evaluation mode
model.eval()

# Initialize variables to track accuracy
correct = 0
total = 0

# No gradient computation is needed during evaluation
with torch.no_grad():
    for images, targets in test_dataloader:
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        outputs = model(images)

        # Get predictions
        _, predicted = torch.max(outputs, 1)

        # Update metrics
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

# Calculate accuracy
accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%")
