import os
import json
from pathlib import Path

import torch
import torch.nn as nn

MODEL_DIR = Path("models") / "RNN" / "QiXiang_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VOCAB_PATH = MODEL_DIR / "vocab.json"
CHECKPOINT_PATH = MODEL_DIR / "best.pt"

MAX_LEN = 20

class IntentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class IntentLSTM_Dropout(IntentLSTM):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__(vocab_size, embed_dim, hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h_last = self.dropout(h[-1])
        return self.fc(h_last)


class IntentBiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        h_forward = h[-2]
        h_backward = h[-1]
        h_cat = torch.cat((h_forward, h_backward), dim=1)
        h_cat = self.dropout(h_cat)
        return self.fc(h_cat)


# Helpers

def save_vocab(vocab: dict, path: Path = VOCAB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)


def load_vocab(path: Path = VOCAB_PATH):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(model: nn.Module, path: Path = CHECKPOINT_PATH, device=None):
    if not Path(path).exists():
        return False
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    return True


def get_model(model_type: str, **kwargs):
    model_type = model_type.lower()
    if model_type == "lstm":
        return IntentLSTM(**kwargs)
    if model_type == "lstm_dropout":
        return IntentLSTM_Dropout(**kwargs)
    if model_type == "bilstm":
        return IntentBiLSTM(**kwargs)
    raise ValueError("Unknown model type: %s" % model_type)
