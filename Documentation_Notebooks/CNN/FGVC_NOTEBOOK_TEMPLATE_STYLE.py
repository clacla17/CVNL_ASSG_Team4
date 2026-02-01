"""
FGVC Aircraft Classification - Notebook Template

Adapted to match your CNN project coding style from school activities.
Uses simple, direct code with explanatory comments.
"""

# ============================================================================
# Cell 1: Imports
# ============================================================================
# Import system and path utilities
import sys
sys.path.append('./src')

# PyTorch and torchvision
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Transforms and models
from torchvision import transforms, models

# Utilities
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Metrics and custom loader
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support,\
                           confusion_matrix, classification_report
from CNN.fgvc_dataset_loader import get_fgvc_loaders

print("Imports successful")


# ============================================================================
# Cell 2: Device and Hyperparameters
# ============================================================================
# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using GPU: {}".format(torch.cuda.get_device_name(0)))
else:
    device = torch.device("cpu")
    print("Using CPU")

# Hyperparameters - using your school notebook naming convention
B = 32  # Batch size
C = 3   # Input channels (RGB)
IMG_SIZE = 224  # Image size after resize
K = 3   # Kernel size for convolutions

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================================
# Cell 3: Load FGVC Dataset
# ============================================================================
# Define transforms for training and validation
# Training: Apply augmentation to improve generalization
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05),\
                           scale=(0.98, 1.02)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

# Validation/Test: No augmentation, just normalize
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# Load FGVC dataset using official splits
dataset_root = './data/CV/aircraft'
loaders_dict = get_fgvc_loaders(
    dataset_root,
    batch_size=B,
    num_workers=2,
    train_transform=train_transform,
    val_transform=test_transform
)

# Extract loaders and class information
train_loader = loaders_dict['train']
val_loader = loaders_dict['val']
test_loader = loaders_dict['test']
num_classes = loaders_dict['num_classes']
idx_to_label = loaders_dict['idx_to_label']

print("\nDataset loaded successfully!")
print("Training batches: {}".format(len(train_loader)))
print("Validation batches: {}".format(len(val_loader)))
print("Test batches: {}".format(len(test_loader)))
print("Number of aircraft models: {}".format(num_classes))


# ============================================================================
# Cell 4: Visualize Sample Images (Optional)
# ============================================================================
# Get one batch to visualize
images, labels = next(iter(train_loader))

# Helper function to denormalize tensor for visualization
def denormalize_batch(images):
    # Images come out of transforms as [0, 1]
    # For visualization, convert to numpy and permute dimensions
    return images.permute(0, 2, 3, 1).numpy()

# Plot 8 sample images
f, axarr = plt.subplots(2, 4, figsize=(12, 6))
for count in range(8):
    row = count // 4
    col = count % 4
    
    # Get image and denormalize
    img = denormalize_batch(images[count:count+1])[0]
    label_idx = labels[count].item()
    model_name = idx_to_label[label_idx]
    
    axarr[row, col].imshow(img)
    axarr[row, col].set_title(model_name)
    axarr[row, col].axis('off')

plt.tight_layout()
plt.show()


# ============================================================================
# Cell 5: Define Model Architecture
# ============================================================================
# Simple CNN architecture - similar to your school notebooks but adapted for FGVC
# This is Iteration 1: Baseline model

filters = 16  # Number of filters in first conv layer

model_cnn_baseline = nn.Sequential(
    # First conv block - learn basic features
    nn.Conv2d(C, filters, K, padding=K//2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    # Second conv block - learn more complex features
    nn.Conv2d(filters, filters*2, K, padding=K//2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    # Third conv block
    nn.Conv2d(filters*2, filters*4, K, padding=K//2),
    nn.ReLU(),
    nn.MaxPool2d(2),
    
    # Flatten and classify
    # After 3 pooling layers: 224 -> 112 -> 56 -> 28
    nn.Flatten(),
    nn.Linear(filters*4 * 28 * 28, 512),
    nn.ReLU(),
    nn.Linear(512, num_classes),
)

print("Model architecture:")
print(model_cnn_baseline)
print("\nTotal parameters: {:,}".format(
    sum(p.numel() for p in model_cnn_baseline.parameters())))


# ============================================================================
# Cell 6: Training Loop (Iteration 1)
# ============================================================================
# Move model to device
model = model_cnn_baseline.to(device)

# Define loss function and optimizer
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training parameters
epochs = 15

# Track training history
train_losses = []
val_losses = []
val_accs = []

print("Starting training on {}".format(device))
print("Epochs: {}, Batch size: {}".format(epochs, B))
print("-" * 60)

for epoch in range(epochs):
    # ===== Training Phase =====
    model.train()
    running_loss = 0.0
    
    for images, labels in train_loader:
        # Move to device
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        logits = model(images)
        loss = loss_func(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    train_loss = running_loss / len(train_loader.dataset)
    
    # ===== Validation Phase =====
    model.eval()
    val_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            logits = model(images)
            loss = loss_func(logits, labels)
            val_loss += loss.item() * images.size(0)
            
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    
    # Store history
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print("Epoch {:2d}: Train Loss = {:.4f}, Val Loss = {:.4f}, Val Acc = {:.4f}".format(
            epoch+1, train_loss, val_loss, val_acc))

print("-" * 60)
print("Training complete!")


# ============================================================================
# Cell 7: Plot Training Results
# ============================================================================
# Plot loss curves
f, axarr = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
axarr[0].plot(train_losses, marker='o', label='Training Loss')
axarr[0].plot(val_losses, marker='s', label='Validation Loss')
axarr[0].set_xlabel('Epoch')
axarr[0].set_ylabel('Loss')
axarr[0].set_title('Training and Validation Loss')
axarr[0].legend()
axarr[0].grid(True, alpha=0.3)

# Accuracy plot
axarr[1].plot(val_accs, marker='o', color='green')
axarr[1].set_xlabel('Epoch')
axarr[1].set_ylabel('Accuracy')
axarr[1].set_title('Validation Accuracy')
axarr[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# ============================================================================
# Cell 8: Evaluate on Test Set
# ============================================================================
model.eval()
test_preds = []
test_labels = []
test_loss = 0.0

print("Evaluating on test set...")

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        logits = model(images)
        loss = loss_func(logits, labels)
        test_loss += loss.item() * images.size(0)
        
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

test_loss = test_loss / len(test_loader.dataset)
test_acc = accuracy_score(test_labels, test_preds)

print("-" * 60)
print("Test Set Results:")
print("Test Loss: {:.4f}".format(test_loss))
print("Test Accuracy: {:.4f} ({:.2f}%)".format(test_acc, test_acc * 100))
print("-" * 60)


# ============================================================================
# Cell 9: Detailed Metrics and Confusion Matrix
# ============================================================================
# Per-class metrics
precision, recall, f1, support = precision_recall_fscore_support(
    test_labels, test_preds, average=None, labels=range(num_classes))

print("\nPer-Class Performance:")
print("{:<25} {:<10} {:<10} {:<10}".format("Aircraft Model", "Precision", "Recall", "F1-Score"))
print("-" * 60)
for idx in range(min(num_classes, 10)):  # Show first 10 classes
    model_name = idx_to_label[idx]
    print("{:<25} {:<10.4f} {:<10.4f} {:<10.4f}".format(
        model_name[:25], precision[idx], recall[idx], f1[idx]))

# Plot confusion matrix for selected classes (showing top confusion cases)
cm = confusion_matrix(test_labels, test_preds)

# Only plot confusion matrix if num_classes is reasonable
if num_classes <= 20:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=[idx_to_label[i][:5] for i in range(num_classes)],\
                yticklabels=[idx_to_label[i][:5] for i in range(num_classes)],\
                annot=False, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()


# ============================================================================
# Cell 10: Save Model (Optional)
# ============================================================================
# Save the trained model
model_save_path = './models/CNN/aircraft_cnn_baseline.pth'

torch.save({
    'model_state_dict': model.state_dict(),
    'num_classes': num_classes,
    'idx_to_label': idx_to_label,
    'test_accuracy': test_acc,
    'test_loss': test_loss,
}, model_save_path)

print("Model saved to: {}".format(model_save_path))
