import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes=1):
        super().__init__()

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25)
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, bag):
        if bag.dim() == 3:                   # shape (1, N, D)
            bag = bag.squeeze(0)             # -> (N, D)

        H = self.embedding(bag)              # (N, hidden)
        A = self.attention(H)                # (N, 1)
        A = torch.softmax(A, dim=0)          # weights

        M = torch.sum(A * H, dim=0)          # (hidden,)
        out = self.classifier(M)             # (1,) raw logit

        return out, A.squeeze()
