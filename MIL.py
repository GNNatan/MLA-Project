import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class AttentionMIL(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_classes=1):
        super().__init__()

        self.feature_extractor = nn.Sequential(
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
        H = self.feature_extractor(bag)

        A = self.attention(H)
        A = torch.softmax(A, dim=0)

        M = torch.sum(A*H, dim=0)

        out = self.classifier(M)
        return out, A