import argparse, os
from PIL import Image, ImageDraw

import numpy as np
from tqdm import tqdm

from geometry import get_polygon, is_inside, index_to_coords
from utils import tile_number, tile_x, tile_y

from tiatoolbox.wsicore.wsireader import WSIReader

def is_background(tile, thr = 0.8) -> bool:
    return (tile.astype("float32")/255.).mean() > thr

def label_preview(slide_name, tile_size=None):    
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
    img.save(f"tiles/{slide_name}/overview_with_labels.png")

def extract_labels():
    slide_names = [str(i) for i in range(1, 25)]
    tile_size = None
    for slide_name in tqdm(slide_names, position=0, desc="Extracting labels"):
        labels = list()
        tiles_folder = f"tiles/{slide_name}"
        poly_path = f"data/{slide_name}.xml"
        region = get_polygon(poly_path)
        tiles = os.listdir(tiles_folder)
        tiles = sorted([t for t in tiles if t.startswith("tile")], key = tile_number)
        for tile_name in tqdm(tiles, position=1, desc=f"Slide {slide_name}", leave=False):
            tile_idx  = tile_number(tile_name)
            if tile_size is None:
                tile = Image.open(os.path.join(tiles_folder, tile_name))
                tile_size = tile.size
            x, y = index_to_coords(tile_idx, slide_name, tile_size = tile_size)
            label = int(is_inside(x, y, region))
            labels.append(label)
        np.save(f"tiles/{slide_name}/labels.npy", labels)
        label_preview(slide_name)


def extract_tiles():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_path", required=True, help="Path to your .svs")
    ap.add_argument("--out_dir", default="out", help="Path to output directory")
    ap.add_argument("--level", type=int, default=0, help="Pyramid level (0 = max resolution)")
    ap.add_argument("--tile_size", type=int, default=256, help="tile size (in pixels)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Open WSI

    reader = WSIReader.open(args.wsi_path)
    w, h = reader.info.level_dimensions[args.level]

    print(f"WSI Opened: {args.wsi_path}")
    print(f"Level = {args.level} -> dimensions: [{w}x{h}]")

    xs = list(range(0, w, args.tile_size))
    ys = list(range(0, h, args.tile_size))

    coords = [(x, y) for y in ys for x in xs]

    kept = []
    saved = 0
    for (x, y) in tqdm(coords):
        tile = reader.read_rect(
            location=(x, y),
            size = (args.tile_size, args.tile_size),
            level = args.level,
            interpolation="antialias"
        )

        if is_background(tile):
            continue


        kept.append((x,y))
        Image.fromarray(tile).save(os.path.join(args.out_dir, f"tile_{saved}_x{x}_y{y}.png"))
        saved += 1

    print(f"Sampled {len(coords)} tiles. Kept {len(kept)} tiles.")

    target_width = 1024
    scale = target_width / w

    thumb = reader.read_bounds(
    (0, 0, w, h),
    resolution=scale,
    units="baseline")
    img = Image.fromarray(thumb)
    draw = ImageDraw.Draw(img)

    for (x, y) in kept:
        rect = [int(x * scale), int(y * scale),
                int((x + args.tile_size) * scale), int((y + args.tile_size) * scale)]
        draw.rectangle(rect, outline=(255,0,0), width=2)
    save_path = os.path.join(args.out_dir, "overview_with_tiles.png")
    img.save(save_path)
    print(f"Preview saved to {save_path}")



if __name__ == "__main__":
    extract_labels()