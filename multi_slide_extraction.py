import subprocess

import sys
import os

if __name__ == "__main__":
        for slide in range(1, 25):
                source_dir = f".\data\{slide}.svs"
                out_dir = f".\\tiles\{slide}"
                if os.path.exists(out_dir):
                        print(f"Skipped slide {slide} because output already exists")
                        continue
                command = [sys.executable,
                "tile_extractor.py",
                "--wsi_path", source_dir,
                "--out_dir", out_dir]
                subprocess.run(command, capture_output=False, text=False)