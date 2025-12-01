import re
import numpy as np

from torchvision import transforms


def tile_number(file_name):
    n = file_name.split("_")[1]
    return int(n)


def tile_x(file_name):
    x = file_name.split("_")[2]
    return int(re.sub(r"[a-zA-Z]", "", x))


def tile_y(file_name):
    y = file_name.split("_")[3]
    return int(re.sub(r"[a-zA-Z.]", "", y))

def normalize(x):
    x = np.array(x, dtype=np.float64)
    return (x - np.min(x)) / (np.max(x) - np.min(x))


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])