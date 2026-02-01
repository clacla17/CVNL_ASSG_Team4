"""
Aircraft Family Classification Model
====================================

RegularizedCNN - Best performing model for aircraft family recognition
Trained on FGVC-Aircraft dataset with 5 target families.

Performance:
- Test Accuracy: 72.0%
- Weighted F1-Score: 72.2%
- Training: 25 epochs with class weights and Adam optimizer

Author: Claire
Date: February 2026
"""

import torch
import torch.nn as nn
from pathlib import Path


class RegularizedCNN(nn.Module):
    """
    5-layer Convolutional Neural Network with Batch Normalization and Dropout.
    
    Architecture:
    - 5 convolutional blocks (32→64→128→256→256 filters)
    - Batch Normalization after each conv layer for training stability
    - MaxPooling for spatial dimension reduction
    - Dropout (0.4) in classifier for regularization
    - 2 fully connected layers for classification
    
    Input: RGB images of size 224x224
    Output: Logits for 5 aircraft families (or num_classes)
    """
    
    def __init__(self, num_classes=5):
        super().__init__()
        
        self.features = nn.Sequential(
            # Conv block 1: 3→32 (224x224 → 112x112)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 2: 32→64 (112x112 → 56x56)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 3: 64→128 (56x56 → 28x28)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 4: 128→256 (28x28 → 14x14)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Conv block 5: 256→256 (14x14 → 14x14, no pooling)
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Moderate dropout to balance regularization and learning capacity
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
            
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        x = self.features(x)         # Extract features: (B, 256, 14, 14)
        x = x.view(x.size(0), -1)    # Flatten: (B, 256*14*14)
        x = self.classifier(x)       # Classify: (B, num_classes)
        return x


def load_model(checkpoint_path, device='cpu', num_classes=5):
    """
    Load a trained RegularizedCNN model from checkpoint.
    
    Args:
        checkpoint_path (str or Path): Path to the .pt checkpoint file
        device (str): Device to load model on ('cpu' or 'cuda')
        num_classes (int): Number of output classes
        
    Returns:
        tuple: (model, checkpoint_dict) where checkpoint_dict contains metadata
        
    Example:
        >>> model, checkpoint = load_model('models/CNN/claire_model.pt', device='cuda')
        >>> model.eval()
        >>> # Use model for inference
    """
    from pathlib import Path
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get num_classes from checkpoint if available
    if 'num_classes' in checkpoint:
        num_classes = checkpoint['num_classes']
    
    # Create model and load weights
    model = RegularizedCNN(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    if 'test_accuracy' in checkpoint:
        print(f"  Test Accuracy: {checkpoint['test_accuracy']:.2%}")
    if 'test_f1' in checkpoint:
        print(f"  Test F1-Score: {checkpoint['test_f1']:.2%}")
    if 'epoch' in checkpoint:
        print(f"  Trained Epochs: {checkpoint['epoch']}")
    
    return model, checkpoint


def get_model_info(model):
    """
    Print model architecture and parameter count.
    
    Args:
        model (nn.Module): PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (float32)")


# Aircraft family names (default for FGVC-Aircraft 5-family classification)
DEFAULT_CLASS_NAMES = [
    'A320_Family',
    'A330_Family', 
    'B737_Family',
    'B777_Family',
    'B787_Family'
]


if __name__ == "__main__":
    # Demo: Create and inspect model
    print("Creating RegularizedCNN model...")
    model = RegularizedCNN(num_classes=5)
    get_model_info(model)
    
    # Demo: Forward pass
    print("\nTesting forward pass...")
    model.eval()  # Set to eval mode (required for BatchNorm with batch_size=1)
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():  # No gradients needed for demo
        output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output}")
    
    # Demo: Load trained model (if checkpoint exists)
    checkpoint_path = Path(__file__).parent.parent.parent.parent / 'models' / 'CNN' / 'claire_model.pt'
    if checkpoint_path.exists():
        print(f"\n{'='*60}")
        print("Loading trained model...")
        print(f"{'='*60}")
        trained_model, checkpoint = load_model(checkpoint_path, device='cpu')
        
        if 'class_names' in checkpoint:
            print(f"Class names: {checkpoint['class_names']}")
    else:
        print(f"\nCheckpoint not found at: {checkpoint_path}")
        print("Train the model first using the notebook!")
