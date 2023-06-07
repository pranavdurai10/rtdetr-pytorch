'''
///////////////////////////////////////////////////////////////////////////
Code written by Pranav Durai on 06.06.2023 @ 22:00:56

About: Training Script to train RTDeTR-L Model

Framework: PyTorch 2.0
///////////////////////////////////////////////////////////////////////////
'''

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import DataLoader
from models.rtdetr_l import RTDeTRL


# Set device: NVIDIA CUDA (or) CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
num_classes = 80
scales = {'l': [1.00, 1.00, 1024]}
lr = 0.001
batch_size = 16
num_epochs = 10

# Create the RTDETR model
model = RTDeTRL(num_classes=num_classes, scales=scales)
model.to(device)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Load and preprocess the dataset
transform = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = "PATH_TO_DATASET"
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress
        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item()}')

# Save the trained RTDETR model
torch.save(model.state_dict(), 'models/rtdetr-l.pth')