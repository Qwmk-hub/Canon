import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from model.mobilenetv2 import get_mobilenet
from model.efficientnet import get_efficientnet_b1, get_efficientnet_b2

from utils.train_utils import train_one_epoch, evaluate, save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration
num_classes = 4
batch_size = 128
epochs = 8
lr = 1e-3

train_dir = 'data/4class/train'
val_dir = 'data/4class/val'
model_save_path = 'outputs/best_model.pth'

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Datasets and loaders
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
print("📌 Class index:", train_dataset.class_to_idx)
val_dataset = datasets.ImageFolder(val_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Model
# model = get_mobilenet(num_classes=num_classes).to(device)
model = get_efficientnet_b1(num_classes=num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
best_acc = 0.0
for epoch in range(epochs):
    # print(f"Epoch [{epoch+1}/{epochs}]")

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        save_model(model, model_save_path)
        print("✅ Best model saved!")

# Save last epoch model
save_model(model, 'outputs/last_epoch_model.pth')
print("💾 Last epoch model saved!")
