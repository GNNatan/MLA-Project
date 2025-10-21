import subprocess

import sys


for slide in range(2, 25):
    command = [sys.executable,
            "tile_extractor.py",
            "--wsi_path", f".\data\{slide}.svs",
            "--out_dir", f".\out\{slide}"]
    subprocess.run(command, capture_output=False, text=False)