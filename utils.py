import re

def tile_number(file_name):
    n = file_name.split("_")[1]
    return int(n)


def tile_x(file_name):
    x = file_name.split("_")[2]
    return int(re.sub(r"[a-zA-Z]", "", x))


def tile_y(file_name):
    y = file_name.split("_")[3]
    return int(re.sub(r"[a-zA-Z.]", "", y))