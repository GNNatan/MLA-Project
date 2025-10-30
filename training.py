import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from MIL import AttentionMIL
from geometry import tumor, is_inside, index_to_coords


train = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

def bags_from_image(slide_name, n_bags = 4):
    bags = []
    labels = []
    path = '' #from slide_name get .npy
    label_path = '' #from slide_name get .xml
    region = tumor(label_path)
    patches = torch.from_numpy(np.load(path))
    n_samples = patches.size(0)
    base_size = n_samples// n_bags
    remainder = n_samples % n_bags
    start = 0
    for i in range(n_bags):
        end = start + base_size
        if (i == n_bags - 1):
            end += remainder
        bag = patches[start:end]
        bags.append(bag)
        label = 0
        while (start != end):
            x, y = index_to_coords(start, slide_name)
            if is_inside(x, y, region):
                label = 1
                start = end
            else:
                start += 1
        labels.append(label)
    return bags, labels

class Training_Set(torch.utils.data.Dataset):
    def __init__(self, train = train, bags_per_image = 4):
        self.bags = []
        self.labels = []
        for t in train:
            b, l = bags_from_image(t, bags_per_image)
            self.bags.extend(b)
            self.labels.extend(l)
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        return self.bags[idx], torch.tensor(self.labels[idx], dtype=torch.float32)


#training

device = "cuda" if torch.cuda.is_available() else "cpu"

input_dim = 2048
hidden_dim = 64
n_epochs = 10
lr = 1e-3

dataset = Training_Set()
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

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
print("Attention weights:", A.squeeze().detach().cpu().numpy())