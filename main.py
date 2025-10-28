from feature_extractor import create_feature_extractor, extract_features
from PIL import Image
import os
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    feat_extractor = create_feature_extractor()
    output_dir = "feats"
    os.makedirs(output_dir, exist_ok=True)
    folders = [f"tiles/{i}" for i in range(17, 25)]
    for folder in folders:
        print(f"Extracting features from {folder}")
        patches = []
        for filename in tqdm(os.listdir(folder)):
            if filename == "overview_with_tiles.png":
                continue
            full_path = os.path.join(folder, filename)
            with Image.open(full_path) as img:
                patch = np.array(img)
                patches.append(patch)
        save_name = folder.split("/")[1]
        feat = extract_features(patches, feat_extractor)
        np.save(f"{os.path.join(output_dir, save_name)}.npy", feat)
                
