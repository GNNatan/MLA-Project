import torch
import torch.nn as nn
import torch.optim as optim


import numpy as np

import os

from PIL import Image

from tqdm import tqdm

from MIL import AttentionMIL

from utils import tile_number, preprocess

DEBUG = True

np.random.seed(42)

checkpoint_path = os.path.join("checkpoints", "attention")

os.makedirs(checkpoint_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

slides_dir = "tiles"

class MultiBagMILDataset(torch.utils.data.Dataset):
    def __init__(self, slides, bag_size = 350, transform = preprocess):
        self.bag_size = bag_size
        self.transform = transform

        self.bags = []
        self.labels = []

        total_pos = 0
        total_neg = 0

        for slide in tqdm(slides, leave=False):
            slide_path = os.path.join(slides_dir, slide)

            if not os.path.isdir(slide_path):
                continue
        
            patch_files = os.listdir(slide_path)
            patch_files = sorted([f for f in patch_files if f.startswith("tile")], key = tile_number)
            patch_labels = np.load(os.path.join(slide_path, "labels.npy"))

            indices = [tile_number(p) for p in patch_files]

            np.random.shuffle(indices)

            pos_idx = [i for i in indices if patch_labels[i] == 1]
            neg_idx = [i for i in indices if patch_labels[i] == 0]

            def generate_bag(force_negative = False):
                if force_negative:
                    if len(neg_idx) >= bag_size:
                        bag_indices = neg_idx[:bag_size]
                        for index in bag_indices:
                            try:
                                indices.remove(index)
                            except ValueError:
                                pass
                        del neg_idx[:bag_size]
                        return bag_indices, 0
                j = min(bag_size, len(indices))
                bag_indices = indices[:j]
                label = 0
                for index in bag_indices:
                    label = max(label, patch_labels[index])
                    try:
                        pos_idx.remove(index)
                    except ValueError:
                        pass
                    try:
                        neg_idx.remove(index)
                    except ValueError:
                        pass
                del indices[:j]
                return bag_indices, label
                
            pos = 0
            neg = 0

            while len(indices) > 0:
                force_negative = pos > neg                
                bag_indices, label = generate_bag(force_negative=force_negative)
                pos += label
                neg += 1 - label
                self.bags.append((slide_path, bag_indices))
                self.labels.append(torch.tensor(label, dtype=torch.float32))

            total_pos += pos
            total_neg += neg

        if DEBUG:
            print(f"Created {total_pos} positive and {total_neg} negative bags. Total: {total_pos+total_neg}")



    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):

        slide_path, indices = self.bags[idx]

        patch_files = os.listdir(slide_path)
        patch_files = sorted([f for f in patch_files if f.startswith("tile")], key = tile_number)

        patches = []

        for i in indices:
            img = Image.open(os.path.join(slide_path, patch_files[i])).convert('RGB')
            if self.transform:
                img = self.transform(img)

            patches.append(img)
        
        patches = torch.stack(patches)

        return patches, self.labels[idx]




def train_one_epoch(model, loader, optimizer):
    model.train()

    train_loss = 0.
    train_err = 0.

    for data, label in tqdm(loader, desc= "Training", leave = False):

        data = data.to(device)
        label = label.float().to(device)

        optimizer.zero_grad()        

        loss, _ = model.calculate_objective(data, label)
        train_loss += loss.item()

        error, _ = model.calculate_classification_error(data, label)

        train_err += error

        loss.backward()
        optimizer.step()

    train_err /= len(loader)
    train_loss /= len(loader)

    return train_loss, train_err


@torch.no_grad()
def validate(model, loader):
    model.eval()

    val_loss = 0.
    val_err = 0.

    for data, label in tqdm(loader, desc= "Validating", leave = False):

        data = data.to(device)
        label = label.float().to(device) 

        loss, _ = model.calculate_objective(data, label)
        val_loss += loss.item()

        error, _ = model.calculate_classification_error(data, label)

        val_err += error

    val_err /= len(loader)
    val_loss /= len(loader)

    return val_loss, val_err

def train_model(model, train_loader, val_loader, epochs=100):

    optimizer = optim.Adam(
    model.parameters(),
    lr = 1e-4,
    weight_decay = 5e-4
    )

    latest_checkpoint = os.path.join(checkpoint_path, f"latest.pth")

    checkpoint_epoch = None

    best_val = None

    if os.path.exists(latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint)
        checkpoint_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val = checkpoint['best_val']

    start_epoch = 0

    if checkpoint_epoch is not None:
        start_epoch = checkpoint_epoch

    epochs = tqdm(range(start_epoch, epochs), desc="Training loop", initial=start_epoch, total=epochs)

    for epoch in epochs:
        train_loss, train_err = train_one_epoch(model, train_loader, optimizer)

        val_loss, val_err = validate(model, val_loader)

        val = val_err + val_loss

        if (epoch + 1) % 20 == 0:
            torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val': best_val}, os.path.join(checkpoint_path, f"epoch_{epoch+1}.pth"))


        if best_val is None or val < best_val:
            best_val = val

            torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val': best_val}, os.path.join(checkpoint_path, "best.pth"))


        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val': best_val}, latest_checkpoint)

        epochs.set_postfix(ordered_dict={"loss": val_loss, "err": val_err,"last": val, "best": best_val})


def main():
    model = AttentionMIL(pooling="attention").to(device)
    
    train_names = [str(i) for i in range(14)]

    val_names = [str(i) for i in range(14, 17)]

    train_dataset = MultiBagMILDataset(train_names)
    val_dataset = MultiBagMILDataset(val_names)

    train_loader = torch.utils.data.DataLoader(train_dataset, num_workers=16)
    val_loader = torch.utils.data.DataLoader(val_dataset, num_workers = 16)

    train_model(model, train_loader, val_loader, 100)

    print("Finished training!")

if __name__ == "__main__":
    main()