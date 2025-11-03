import torch
import math

import numpy as np
from geometry import get_polygon, is_inside, index_to_coords
from tqdm import trange

import os


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
    for _ in trange(n_bags):
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


if __name__ == "__main__":

    bag_size = 256

    slide_names = [str(i) for i in range(17, 25)]
    output_dir = "bags"
    os.makedirs(output_dir, exist_ok=True)
    for slide_name in slide_names:
        out_path = os.path.join(output_dir, slide_name)
        os.makedirs(out_path, exist_ok= True)
        bags, labels = bags_from_image(slide_name, bag_size)
        np.save(f"{os.path.join(out_path, 'bags')}.npy", bags)
        np.save(f"{os.path.join(out_path, 'labels')}.npy", labels)
        