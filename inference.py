import torch
import torch.nn as nn
import torch.nn.functional as F

from tiatoolbox.wsicore.wsireader import WSIReader
from PIL import Image, ImageDraw

from utils import tile_x, tile_y
import numpy as np
import os
from tqdm import tqdm

from utils import tile_number, preprocess

from models import AttentionMIL

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def infer_patch(model, patch):
    model.eval()
    patch = patch.to(device).unsqueeze(0).unsqueeze(0)    # [1,1,3,224,224]
    Y_prob, _, _ = model(patch)              # shape [1,1]
    return Y_prob.item()

def preview(model, slide_name, model_name="attention", tile_size=None):    
    slide_path = f"tiles/{slide_name}"
    files = os.listdir(slide_path)
    files = sorted([f for f in files if f.startswith("tile")], key = tile_number)
    os.makedirs(f"inference/{model_name}", exist_ok=True)

    reader = WSIReader.open(f"data/{slide_name}.svs")
    full_width, full_height = reader.info.level_dimensions[0]

    target_width = 1024
    scale = target_width / full_width

    thumb = reader.read_bounds((0, 0, full_width, full_height),
                               resolution=scale, units="baseline")
    img = Image.fromarray(thumb)
    draw = ImageDraw.Draw(img)

    if tile_size is None:
        first_tile = Image.open(f"tiles/{slide_name}/{files[0]}")
        tile_size = first_tile.size

    tile_width, tile_height = tile_size

    for file in tqdm(files, desc=f"Evaluating slide {slide_name}"):
        patch = preprocess(Image.open(f"tiles/{slide_name}/{file}").convert("RGB"))
        score = infer_patch(model, patch)

        color = (int(255 * score), 0, int(255 * (1 - score)), 64)

        x = tile_x(file)
        y = tile_y(file)

        rect = [
            int(x * scale),
            int(y * scale),
            int((x + tile_width) * scale),
            int((y + tile_height) * scale)
            ]
        draw.rectangle(rect, fill=color)
    img.save(f"inference/{model_name}/{slide_name}.png")

def truth_preview(slide_name, tile_size=None):    
    slide_path = f"tiles/{slide_name}"
    files = os.listdir(slide_path)
    files = sorted([f for f in files if f.startswith("tile")], key = tile_number)
    os.makedirs(f"inference/truth", exist_ok=True)
    labels = np.load(f"{slide_path}/labels.npy")
    reader = WSIReader.open(f"data/{slide_name}.svs")
    full_width, full_height = reader.info.level_dimensions[0]

    target_width = 1024
    scale = target_width / full_width

    thumb = reader.read_bounds((0, 0, full_width, full_height),
                               resolution=scale, units="baseline")
    img = Image.fromarray(thumb)
    draw = ImageDraw.Draw(img)

    if tile_size is None:
        first_tile = Image.open(f"tiles/{slide_name}/{files[0]}")
        tile_size = first_tile.size

    tile_width, tile_height = tile_size

    for index, file in enumerate(tqdm(files, desc=f"Evaluating slide {slide_name}")):
        score = labels[index]

        color = (int(255 * score), 0, int(255 * (1 - score)), 64)

        x = tile_x(file)
        y = tile_y(file)

        rect = [
            int(x * scale),
            int(y * scale),
            int((x + tile_width) * scale),
            int((y + tile_height) * scale)
            ]
        draw.rectangle(rect, fill=color)
    img.save(f"inference/truth/{slide_name}.png")


if __name__ == "__main__":
    test_set = [str(i) for i in range(1, 25)]
    for slide_name in test_set:
        truth_preview(slide_name)
    # for pooling in ["attention", "mean", "max", "attention_balanced", "max_balanced", "mean_balanced"]:
    #  for cp in ["best", "latest"]:
    #     checkpoint_name = f"checkpoints/{pooling}/{cp}.pth"
    #     checkpoint = torch.load(checkpoint_name, map_location=device)

    #     model = AttentionMIL(pooling.split("_")[0])
    #     model.load_state_dict(checkpoint["model_state_dict"])
    #     model = model.to(device)
    
    #     for slide_name in test_set:
    #         preview(model, slide_name, f"{pooling}/{cp}")