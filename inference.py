import torch
import torch.nn as nn
import torch.nn.functional as F

from tiatoolbox.wsicore.wsireader import WSIReader
from PIL import Image, ImageDraw

from utils import tile_x, tile_y
import numpy as np
import os
from tqdm.contrib import tzip

from MIL import AttentionMIL

test_set = [str(i) for i in range(1, 25)]



def preview(scores, slide_name, checkpoint_name, tile_size=None):
    folder = f"tiles/{slide_name}"
    files = os.listdir(folder)
    files = [f for f in files if f.startswith("tile")]

    if len(files) != len(scores):
        print(f"[WARNING] Files={len(files)}, scores={len(scores)} mismatch.")
        return

    os.makedirs(f"inference/{checkpoint_name}", exist_ok=True)

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

    for file, score in tzip(files, scores):
        score = float(score)

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

    img.save(f"inference/{checkpoint_name}/{slide_name}.png")



# Model parameters
input_dim = 2048
hidden_dim = 64

if __name__ == "__main__":
    for checkpoint_name in ["baseline"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        checkpoint_path = f"checkpoints/{checkpoint_name}/mil_model_300.pth"

        model = AttentionMIL(input_dim, hidden_dim)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        model.eval()

        scores = []

        with torch.no_grad():
            for slide_name in test_set:

                bags_npy = np.load(
                    f"bags/{slide_name}/bags.npy", allow_pickle=True
                )
                bags = [torch.from_numpy(np.array(bag, dtype=np.float32)) for bag in bags_npy]

                bag_values = []

                for bag in bags:
                    if bag.dim() == 3:
                        bag = bag.squeeze(0)
                    bag = bag.to(device)

                    out, attn = model(bag)

                    attn_np = attn.cpu().numpy()

                    bag_values.extend(attn_np.tolist())

                score = torch.tensor(bag_values)
                scores.append(score)

        for index, slide_name in enumerate(test_set):
            preview(scores[index], slide_name, checkpoint_name)