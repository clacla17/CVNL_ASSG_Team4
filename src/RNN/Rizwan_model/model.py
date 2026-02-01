"""
Sentiment LSTM Model
Bidirectional LSTM for Twitter Airline Sentiment Analysis

Author: Rizwan
Model: Bidirectional LSTM
"""

import torch
import torch.nn as nn


class SentimentLSTM(nn.Module):
    """
    Bidirectional LSTM for sentiment classification.
    
    Architecture:
    - Embedding layer with padding
    - Multi-layer Bidirectional LSTM
    - Fully connected layers with dropout
    - Three-class classification
    
    Args:
        vocab_size (int): Size of vocabulary
        embedding_dim (int): Dimension of embeddings (default: 100)
        hidden_dim (int): LSTM hidden dimension (default: 128)
        output_dim (int): Number of classes (default: 3)
        n_layers (int): Number of LSTM layers (default: 2)
        bidirectional (bool): Use bidirectional LSTM (default: True)
        dropout (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128,
                 output_dim=3, n_layers=2, bidirectional=True, dropout=0.3):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # FC layers
        self.fc1 = nn.Linear(lstm_output_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
        
        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text):
        """
        Forward pass.
        
        Args:
            text (Tensor): Input tensor (batch_size, seq_len)
        
        Returns:
            Tensor: Output logits (batch_size, output_dim)
        """
        # Embed with dropout
        embedded = self.dropout(self.embedding(text))
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Dropout on hidden states
        hidden = self.dropout(hidden)
        
        # Concatenate bidirectional hidden states
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        else:
            hidden = hidden[-1]
        
        # FC layers
        out = self.fc1(hidden)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


# Config presets for different iterations
MODEL_CONFIGS = {
    'baseline': {
        'embedding_dim': 100,
        'hidden_dim': 128,
        'output_dim': 3,
        'n_layers': 2,
        'bidirectional': True,
        'dropout': 0.3
    },
    'improved': {
        'embedding_dim': 150,
        'hidden_dim': 160,
        'output_dim': 3,
        'n_layers': 3,
        'bidirectional': True,
        'dropout': 0.35
    },
    'final': {
        'embedding_dim': 150,
        'hidden_dim': 160,
        'output_dim': 3,
        'n_layers': 3,
        'bidirectional': True,
        'dropout': 0.4
    }
}


def create_model(vocab_size, config='final', device='cpu'):
    """
    Create a SentimentLSTM model with preset configs.
    
    Args:
        vocab_size (int): Size of vocabulary
        config (str): Config preset ('baseline', 'improved', or 'final')
        device (str): Device to use ('cpu' or 'cuda')
    
    Returns:
        SentimentLSTM: Initialized model
    """
    if config not in MODEL_CONFIGS:
        raise ValueError(f"Config '{config}' not found. Available: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[config]
    model = SentimentLSTM(vocab_size=vocab_size, **model_config)
    model = model.to(device)
    
    return model


if __name__ == "__main__":
    # Example usage
    vocab_size = 10000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with final configuration
    model = create_model(vocab_size, config='final', device=device)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}")
    
    # Test forward pass
    batch_size = 32
    seq_len = 50
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
