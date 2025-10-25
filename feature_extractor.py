import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

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
