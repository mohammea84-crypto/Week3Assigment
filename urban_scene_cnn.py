# ============================================================
# urban_scene_cnn.py
# Neural Network Models for Urban Scene Classification
# MIT Urban Scene Dataset - Subset: 4 classes
# ============================================================

# ── STEP 3: Load and Prepare the Dataset ────────────────────
# Commit: "Loaded and preprocessed MIT Places dataset"

import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# Image transformations: resize, convert to tensor, normalize
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load dataset from local 'data/' folder
dataset_path = "./data"
dataset = ImageFolder(root=dataset_path, transform=transform)
print(f"Classes: {dataset.classes}")
print(f"Total images: {len(dataset)}")

# Split into 70% train, 15% val, 15% test
train_size = int(0.70 * len(dataset))
val_size   = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size
train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])

# DataLoaders
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=32, shuffle=False)

# Save a sample image grid for the presentation
import torchvision.utils as vutils
import numpy as np

def save_sample_grid(dataset, classes, filename="outputs/sample_images.png"):
    os.makedirs("outputs", exist_ok=True)
    fig, axes = plt.subplots(1, len(classes), figsize=(12, 3))
    shown = {c: False for c in range(len(classes))}
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    for img, lbl in dataset:
        if not shown[lbl]:
            ax = axes[lbl]
            # Denormalize
            img_np = img.permute(1, 2, 0).numpy()
            img_np = std * img_np + mean
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)
            ax.set_title(classes[lbl], fontsize=9)
            ax.axis("off")
            shown[lbl] = True
        if all(shown.values()):
            break
    plt.suptitle("Sample Images per Class", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

save_sample_grid(dataset, dataset.classes)


# ── STEP 4: Build the CNN Model ─────────────────────────────
# Commit: "Implemented CNN model for urban scene classification"

import torch.nn as nn
import torch.optim as optim

class UrbanSceneCNN(nn.Module):
    """Simple CNN with BatchNorm and Dropout for urban scene classification."""
    def __init__(self, num_classes):
        super(UrbanSceneCNN, self).__init__()
        # Conv block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        # Conv block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)

        self.relu  = nn.ReLU()
        self.pool  = nn.MaxPool2d(2, 2)
        self.drop  = nn.Dropout(0.4)

        # After 2 pool layers: 128 -> 64 -> 32, channels=64
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 64x64
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 32x32
        x = torch.flatten(x, 1)
        x = self.drop(self.relu(self.fc1(x)))
        return self.fc2(x)

num_classes = len(dataset.classes)
model = UrbanSceneCNN(num_classes)
print(model)

