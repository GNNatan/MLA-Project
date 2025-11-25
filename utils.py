import re
import numpy as np



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