import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyModel(nn.Module):
    def __init__(self, pred=1.):
        self.pred = pred
        super(DummyModel, self).__init__()

    def forward(self, x):
        Y_prob = torch.full((1,1), 1., device=x.device)
        Y_pred = torch.full((1,1), self.pred, device=x.device)
        return Y_prob, Y_pred, None

class AttentionMIL(nn.Module):
    def __init__(self, pooling="attention"):
        super(AttentionMIL, self).__init__()
        self.M = 512
        self.L = 128
        self.ATTENTION_BRANCHES = 1

        self.pooling = "attention" # | "max" | "mean"

        self.feature_extractor_1 = nn.Sequential(
            nn.Conv2d(3, 36, kernel_size = 4),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(36, 48, kernel_size = 3),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((5, 5)),
        )

        self.feature_extractor_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 5 * 5, self.M),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.M, self.M),
            nn.ReLU(),
            nn.Dropout()
        )

        self.attention = nn.Sequential(
            nn.Linear(self.M, self.L),
            nn.Tanh(),
            nn.Linear(self.L, self.ATTENTION_BRANCHES)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.M * self.ATTENTION_BRANCHES, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)

        # feature extraction

        H = self.feature_extractor_1(x)
        H = self.feature_extractor_2(H)  # KxM


        # pooling

        if self.pooling == "attention":
            A = self.attention(H)            # KxATTENTION_BRANCHES
            A = F.softmax(A, dim = 0)

            Z = torch.sum(A * H, dim = 0)
        elif self.pooling == "max":
            Z, _ = torch.max(H, dim = 0)
            A = None
        elif self.pooling == "mean":
            Z = torch.mean(H, dim = 0)
            A = None
        else:
            raise ValueError("pooling must be one of: 'attention', 'max', 'mean'")        
        
        # classification
        Z = Z.unsqueeze(0)

        Y_prob = self.classifier(Z)             # probability
        Y_pred  = torch.ge(Y_prob, 0.5).float() # prediction

        return Y_prob, Y_pred, A
    
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_pred, _ = self.forward(X)
        correct = Y_pred.eq(Y)
        error = 1. - correct.float().mean().item()

        return error, Y_pred

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood.mean(), A