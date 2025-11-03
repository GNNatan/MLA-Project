import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np
from MIL import AttentionMIL
from geometry import get_polygon, is_inside, index_to_coords
from tqdm import tqdm, trange

import os

train = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

def bags_from_image(slide_name, bag_size = 256):
    bags = []
    labels = []
    path = f"feats/{slide_name}.npy"
    label_path = f"data/{slide_name}.xml"
    region = get_polygon(label_path)
    patches = torch.from_numpy(np.load(path))
    n_samples = patches.size(0)
    n_bags = int(math.ceil(n_samples / bag_size))
    start = 0
    for i in trange(n_bags):
        end = min(start + bag_size, n_samples)
        bag = patches[start:end]
        bags.append(bag)
        label = 0
        for patch_idx in range(start, end):
            x, y = index_to_coords(patch_idx, slide_name)
            if is_inside(x, y, region):
                label = 1
                break
        labels.append(label)
        start = end
    return bags, labels

class Training_Set(torch.utils.data.Dataset):
    def __init__(self, train = train, bag_size = 256):
        self.bags = []
        self.labels = []
        print("Initialization...")
        for t in train:
            b, l = bags_from_image(t, bag_size)
            self.bags.extend(b)
            self.labels.extend(l)

    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        return self.bags[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

    def debug(self):
        print(f"{sum(self.labels)} positive bags")


#training
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = 2048
    hidden_dim = 64
    n_epochs = 100
    lr = 1e-3

    dataset = Training_Set(train)
    dataset.debug()
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = AttentionMIL(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    save_dir = 'checkpoints'
    os.makedirs(save_dir, exists_ok = True)
    best_loss = float("inf")

    #resume from checkpoint

    resume_path = None
    start_epoch = 0
    if resume_path and os.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']


    for epoch in trange(start_epoch, n_epochs):
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
        if((epoch + 1)% 20 == 0):
            checkpoint_path = os.path.join(save_dir, f"mil_model_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(dataset)
            }, checkpoint_path)
        if((total_loss / len(dataset)) < best_loss):
            best_loss = total_loss/len(dataset)
            best_path = os.path.join(save_dir, "mil_model_best.pth")
        torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(dataset)
            }, best_path)
        
    print('Finished Training!')