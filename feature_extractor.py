import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

import os
from utils import tile_number

device = "cuda" if torch.cuda.is_available() else "cpu"

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])

def create_feature_extractor():
    base = models.resnet50(pretrained=True)
    feature_extractor = nn.Sequential(*list(base.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor

def extract_features(patches, model):
    features = []
    with torch.no_grad():
        for img in tqdm(patches):
            x = preprocess(Image.fromarray(img)).unsqueeze(0).to(device)
            feat = model(x)
            feat = feat.squeeze().cpu().numpy()
            features.append(feat)
    return np.stack(features)

if __name__ == "__main__":
    feat_extractor = create_feature_extractor()
    output_dir = "feats"
    os.makedirs(output_dir, exist_ok=True)
    folders = [f"tiles/{i}" for i in range(1, 25)]
    for folder in folders:
        print(f"Reading {folder}")
        save_name = folder.split("/")[1]
        out_path = f"{os.path.join(output_dir, save_name)}.npy"
        if os.path.exists(out_path):
            print(f"{out_path} already exists, skipping,,,")
            continue
        patches = []
        files = os.listdir(folder)
        files = [f for f in files if f.startswith("tile")]
        files_sorted = sorted(files, key = tile_number)
        
        for filename in tqdm(files_sorted):
            full_path = os.path.join(folder, filename)
            with Image.open(full_path) as img:
                patch = np.array(img)
                patches.append(patch)
        
        print(f"Extracting features from {folder}")
        feat = extract_features(patches, feat_extractor)
        np.save(out_path, feat)