
import torch
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import timm

# Load the training data
train_images = torch.load("data/processed/train_images.pt")
train_targets = torch.load("data/processed/train_targets.pt")

# Select an image and its target (e.g., the first one)
index = 344  # Change this to view different images
image = train_images[index]
target = train_targets[index]

# Print the shape of the image tensor
print(f"Original image shape: {image.shape}")

# Plot the image
image = image.permute(1, 2, 0)  # Change shape to (256, 256, 3)

plt.imshow(image.numpy(), cmap="gray")  # Use `.numpy()` to convert to NumPy for matplotlib
plt.title(f"Target: {target}")
plt.axis("off")
plt.show()

print(train_targets.unique())


# Load data
train_images = torch.load("data/processed/train_images.pt")
train_images = train_images[0:100]
train_targets = torch.load("data/processed/train_targets.pt")
train_targets = train_targets[0:100]

# Create Dataset and DataLoader
dataset = TensorDataset(train_images, train_targets)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)  # Default batch size = 32

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a timm model
model = timm.create_model("resnet18", pretrained=True, num_classes=4)  # Replace 'resnet18' with any timm model
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # Suitable for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Default learning rate

# Training loop
num_epochs = 5  # You can adjust this
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


with profile(activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("./log/resnet18")) as prof:
    for i in range(10):
        model(images)
        prof.step()

# Print profiling results sorted by self CPU time
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))


# Print profiling results sorted by self CPU memory usage
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cpu_memory_usage", row_limit=30))

prof.export_chrome_trace("trace.json")