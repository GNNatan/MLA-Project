import torch
import math

import numpy as np
from geometry import get_polygon, is_inside, index_to_coords
from tqdm import tqdm

import os


def bags_from_image(slide_name, bag_size = 256):
    bags = []
    labels = []
    path = f"feats/{slide_name}.npy"
    label_path = f"data/{slide_name}.xml"
    region = get_polygon(label_path)
    patches = np.load(path)
    n_samples = patches.shape[0]
    n_bags = int(math.ceil(n_samples / bag_size))
    start = 0
    for _ in tqdm(range(n_bags), position = 1, desc=f"Creating bags for slide {slide_name}", leave=False):
        end = min(start + bag_size, n_samples)
        bag = patches[start:end]
        bags.append(bag)
        label = 0
        for patch_idx in tqdm(range(start, end), position= 2, desc=f"Calculating label", leave=False):
            x, y = index_to_coords(patch_idx, slide_name)
            if is_inside(x, y, region):
                label = 1
                break
        labels.append(label)
        start = end
    return bags, labels


if __name__ == "__main__":

    bag_size = 256

    slide_names = [str(i) for i in range(1, 25)]
    output_dir = "bags"
    os.makedirs(output_dir, exist_ok=True)
    for slide_name in tqdm(slide_names, position=0, desc="Generating bags from slides"):
        out_path = os.path.join(output_dir, slide_name)
        if os.path.isdir(out_path):
            continue
        bags, labels = bags_from_image(slide_name, bag_size)
        os.makedirs(out_path, exist_ok= True)
        np.save(f"{os.path.join(out_path, 'bags')}.npy", bags)
        np.save(f"{os.path.join(out_path, 'labels')}.npy", labels)
        