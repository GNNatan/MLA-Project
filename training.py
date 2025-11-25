import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

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
    labels = [torch.from_numpy(np.array(label, dtype=np.float32)) for label in labels_npy]

    return bags, labels


class Training_Set(torch.utils.data.Dataset):
    def __init__(self, train=train):
        self.bags = []
        self.labels = []
        self.train = train
        for t in tqdm(train, position=0, desc="Initialization", leave=False):
            b, l = get_bags(t)
            self.bags.extend(b)
            self.labels.extend(l)

    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        return self.bags[idx], self.labels[idx].float()

    def label_count(self):
        negative = 0
        positive = 0
        for labels in self.labels:
            label = torch.max(labels)
            negative += int(1 - label.item())
            positive += int(label.item())

        return negative, positive
    
    def balance(self, bag_size=256, seed=0):
        n_neg, n_pos = self.label_count()
        bags_npy, labels_npy = artificial_bags(self.train, n_neg, n_pos, bag_size, seed)
        bags_torch = [torch.from_numpy(np.array(bag, dtype=np.float32)) for bag in bags_npy]

        labels_torch = [torch.from_numpy(np.array(label, dtype=np.float32)) for label in labels_npy]

        self.bags.extend(bags_torch)
        self.labels.extend(labels_torch)



def normalize_labels(label):
    if label.sum() > 0:
        return label / label.sum()
    else:
        return torch.zeros_like(label)


def pool_labels(label):
    return label.max()


# training
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = 2048
    hidden_dim = 64
    n_epochs = 300
    lr = 1e-3
    coeff = 1.

    dataset = Training_Set(train)

    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = AttentionMIL(input_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_bag = nn.BCEWithLogitsLoss()
    criterion_instance = nn.MSELoss()

    save_dir = os.path.join("checkpoints", "baseline")
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float("inf")

    resume_path = None
    start_epoch = 0

    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    epochs = tqdm(range(start_epoch, n_epochs), position=0, desc="Training")

    model.train()
    for epoch in epochs:
        total_loss = 0.0
        correct = 0

        for bag, label in loader:
            bag = bag.squeeze(0).to(device)
            label = label.squeeze(0).to(device)

            output, attn = model(bag)

            label_pool = pool_labels(label.squeeze())
            label_pool = label_pool.view(1).to(device)

            label_norm = normalize_labels(label.squeeze())

            bag_loss = criterion_bag(output.view_as(label_pool), label_pool)

            if coeff < 1.0:
                attn_vec = attn.squeeze()
                instance_loss = criterion_instance(attn_vec, label_norm)
            else:
                instance_loss = torch.tensor(0.0, device=device)

            loss = coeff * bag_loss + (1.0 - coeff) * instance_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                prob = torch.sigmoid(output.view(-1))
                pred = (prob > 0.5).long()
                true_label = label_pool.long()
                correct += int(pred.item() == true_label.item())

            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / len(loader)

        epochs.set_postfix(loss=avg_loss, acc=accuracy)

        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(save_dir, f"mil_model_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, checkpoint_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(save_dir, "mil_model_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, best_path)
        
    print('Finished Training!')