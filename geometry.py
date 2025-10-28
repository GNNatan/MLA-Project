import os
from PIL import Image, ImageDraw
import re
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
from tiatoolbox.wsicore.wsireader import WSIReader
from tqdm import trange

def tumor(file):
    vertices = []
    tree = ET.parse(file)
    root = tree.getroot()

    for annot in root.findall(".//Annotation"):
        for region in annot.findall(".//Region"):
            for v in region.findall(".//Vertex"):
                x = float(v.get("X"))
                y = float(v.get("Y"))
                vertices.append((round(x), round(y)))
    return Polygon(vertices)


def is_inside(x, y, polygon):
    point = Point(x, y)
    return polygon.contains(point) or polygon.touches(point)


def index_to_coords(index:int, slide_name = "1", center = True):
    tiles_path = f"tiles/{slide_name}"
    tiles = os.listdir(tiles_path)
    index += 1 # skip overview
    tile_name = tiles[index]
    x, y = tile_name.split("_")[2:]
    x = int(re.sub(r"[a-zA-Z]", "", x))
    y = int(re.sub(r"[a-zA-Z.]", "", y))
    if center:
        tile = Image.open(os.path.join(tiles_path, tile_name))
        w, h = tile.size
        x += int(w/2)
        y += int(h/2)
    return x, y
    
def preview(slide_name = "1", tile_size = (256, 256)):
    w, h = tile_size
    tiles_path = f"tiles/{slide_name}"
    tiles = os.listdir(tiles_path)
    indices = len(tiles) - 1
    preview = Image.open(os.path.join(tiles_path, "overview_with_tiles.png"))
    draw = ImageDraw.Draw(preview)
    poly = tumor(open(f"data/{slide_name}.xml"))
    reader = WSIReader.open(f"data/{slide_name}.svs")
    W, _ = reader.info.level_dimensions[0]
    scale = 1024 / W
    for index in trange(indices):
        x, y = index_to_coords(index, slide_name=slide_name, center = False)
        cx = x + int(w/2)
        cy = y + int(h/2)
        is_tumor = is_inside(cx, cy, poly)
        color = (255,0,0) if is_tumor else (0, 0, 255)
        rect  = [
            int(x * scale), int(y * scale),
            int((x + w) * scale), int((y + h) * scale)
        ]
        draw.rectangle(rect, outline=color, width=2)

    preview.show()





if __name__ == "__main__":
    preview()