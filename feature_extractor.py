import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def create_feature_extractor():
    base = models.resnet50(pretrained=True)
    feature_extractor = nn.Sequential(*list(base.children())[:-1])
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()
    return feature_extractor

def extract_features(patches, model):
    features = []
    with torch.no_grad():
        for img in patches:
            x = Image.fromarray(img).unsqueeze(0).to(device)
            feat = model(x)
            feat = feat.squeeze().cpu().numpy()
            features.append(feat)
    return np.stack(features)
