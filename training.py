import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import numpy as np
from MIL import AttentionMIL
from tqdm import tqdm

from bag_extractor import artificial_bags

import os

train = [str(i) for i in range(1, 19)]

def get_bags(slide_name):
    bags_npy   = np.load(f"bags/{slide_name}/bags.npy", allow_pickle=True)
    labels_npy = np.load(f"bags/{slide_name}/labels.npy", allow_pickle=True)

    bags = [torch.from_numpy(np.array(bag, dtype=np.float32)) for bag in bags_npy]
    labels = torch.from_numpy(np.array(labels_npy, dtype=np.float32))


    return bags, labels


class Training_Set(torch.utils.data.Dataset):
    def __init__(self, train = train):
        self.bags = []
        self.labels = []
        self.train = train
        for t in tqdm(train,position=0, desc="Initialization", leave=False):
            b, l = get_bags(t)
            self.bags.extend(b)
            self.labels.extend(l)

    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        return self.bags[idx], torch.tensor(self.labels[idx], dtype=torch.float32)

    def label_count(self):
        negative = 0
        positive = 0
        for label in self.labels:
            negative += int(1 - label.item())
            positive += int(label.item())

        return negative, positive
    
    def balance(self, bag_size = 256, seed = 0):
        n_neg, n_pos = self.label_count()
        bags_npy, labels_npy = artificial_bags(self.train, n_neg, n_pos, bag_size, seed)
        bags_torch = [torch.from_numpy(np.array(bag, dtype=np.float32)) for bag in bags_npy]
        labels_torch = torch.from_numpy(np.array(labels_npy, dtype=np.float32))
        self.bags.extend(bags_torch)
        self.labels.extend(labels_torch)




#training
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = 2048
    hidden_dim = 64
    n_epochs = 300
    lr = 1e-3

    dataset = Training_Set(train)
    neg, pos = dataset.label_count()
    dataset.balance()               # balancing dataset by adding artificial bags
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = AttentionMIL(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor(neg/pos).to(device)
#    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) #add penalty to deal with unbalanced dataset
    criterion = nn.BCEWithLogitsLoss()

    save_dir = 'checkpoints\\balanced'
    os.makedirs(save_dir, exist_ok = True)
    best_loss = float("inf")

    #resume from checkpoint

    resume_path = None
    start_epoch = 0
    if resume_path and os.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location = device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']


    for epoch in tqdm(range(start_epoch, n_epochs), position=0, desc="Training"):
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
        tqdm.write(f"Epoch {epoch+1}: loss={total_loss/len(dataset):.4f}, acc={acc:.2f}")
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