# made to test MIL

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

import numpy as np

from models import AttentionMIL

class Dummy(torch.utils.data.Dataset):
    def __init__(self, n_bags=200, n_instances=10, input_dim=32):
        self.bags = []
        self.labels = []
        for _ in range(n_bags):
            bag = torch.randn(n_instances, input_dim)
            if random.random() > 0.5:
                bag[torch.randint(0, n_instances, (1,))] += 2.0
                label = 1
            else:
                label = 0
            self.bags.append(bag)
            self.labels.append(label)

    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        return self.bags[idx], torch.tensor(self.labels[idx], dtype=torch.float32)
    
    def debug(self):
        tensor = self.bags[1]
        print(tensor.size(0))

#training

device = "cuda" if torch.cuda.is_available() else "cpu"

input_dim = 32
hidden_dim = 64
n_epochs = 10
lr = 1e-3

dataset = Dummy()
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
dataset.debug()

model = AttentionMIL(input_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(n_epochs):
    total_loss = 0
    correct = 0
    for bag, label in loader:
        bag, label = bag[0].to(device), label.to(device)
        output, attn = model(bag)
        loss = criterion(output.view(-1), label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.sigmoid(output).round()
        correct += (pred==label).sum().item()
        total_loss += loss.item()
    acc = correct/ len(dataset)
    print(f"Epoch {epoch+1}: loss={total_loss/len(dataset):.4f}, acc={acc:.2f}")

bag, label = dataset[0]
_, A = model(bag.to(device))
a = A.squeeze().detach().cpu().numpy()
print("Attention weights:", a)
a.size()