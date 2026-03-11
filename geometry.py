import os
from PIL import Image, ImageDraw
from utils import tile_number, tile_x, tile_y
import xml.etree.ElementTree as ET
from shapely.geometry import Point, Polygon
from tiatoolbox.wsicore.wsireader import WSIReader
from tqdm import trange

def get_polygon(file):
    polygons = []
    tree = ET.parse(file)
    root = tree.getroot()

    for annot in root.findall(".//Annotation"):
        for region in annot.findall(".//Region"):
            vertices = []
            for v in region.findall(".//Vertex"):
                x = float(v.get("X"))
                y = float(v.get("Y"))
                vertices.append((round(x), round(y)))
            polygons.append(Polygon(vertices))
    return polygons


def is_inside(x, y, polygons):
    point = Point(x, y)
    for polygon in polygons:
        if polygon.contains(point) or polygon.touches(point):
            return True
    return False


def index_to_coords(index:int, slide_name = "1", center = True, tile_size = None):
    tiles_path = f"tiles/{slide_name}"
    tiles = os.listdir(tiles_path)
    tiles = [t for t in tiles if t.startswith("tile")]
    tiles = sorted(tiles, key = tile_number)
    try:
        tile_name = tiles[index]
    except IndexError:
        raise IndexError(f"{index} is out of range for list of size {len(tiles)}")
    x = tile_x(tile_name)
    y = tile_y(tile_name)
    if center:
        if tile_size is None:
            tile = Image.open(os.path.join(tiles_path, tile_name))
            tile_size = tile.size
        w, h = tile_size
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
    poly = get_polygon(open(f"data/{slide_name}.xml"))
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