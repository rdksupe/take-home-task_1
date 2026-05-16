import os
import pandas as pd
import numpy as np
import shutil
import math
from sklearn.model_selection import KFold

# --- CONFIGURATION ---
CSV_PATH = 'annotations.csv'
IMAGE_DIR = 'images'
BASE_DIR = 'datasets_pose'
IMG_W, IMG_H = 640, 480
N_FOLDS = 5
LEVER_ARM = 40  # 40px lever arm for stable angular direction

# Cleanup
if os.path.exists(BASE_DIR):
    shutil.rmtree(BASE_DIR)
os.makedirs(BASE_DIR)

# Load Annotations
df = pd.read_csv(CSV_PATH)
all_images = sorted(df['image'].unique())

# 5-Fold Split
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
splits = list(kf.split(all_images))

for fold, (train_idx, val_idx) in enumerate(splits):
    fold_dir = os.path.join(BASE_DIR, f'fold_{fold}')
    for s in ['train', 'val']:
        os.makedirs(os.path.join(fold_dir, 'images', s), exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'labels', s), exist_ok=True)
    
    for split_name, indices in [('train', train_idx), ('val', val_idx)]:
        for idx in indices:
            image_name = all_images[idx]
            shutil.copy(os.path.join(IMAGE_DIR, image_name), os.path.join(fold_dir, 'images', split_name, image_name))
            
            rows = df[df['image'] == image_name]
            label_name = image_name.replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
            label_path = os.path.join(fold_dir, 'labels', split_name, label_name)
            
            with open(label_path, 'w') as f:
                for _, row in rows.iterrows():
                    bx = (row['bbox_x'] + row['bbox_w']/2) / IMG_W
                    by = (row['bbox_y'] + row['bbox_h']/2) / IMG_H
                    bw = row['bbox_w'] / IMG_W
                    bh = row['bbox_h'] / IMG_H
                    
                    # Point A: Center
                    k1x, k1y = row['center_x'] / IMG_W, row['center_y'] / IMG_H
                    
                    # Point B: Lever Arm Tip
                    rad = np.radians(row['angle_deg'])
                    k2x = (row['center_x'] + LEVER_ARM * np.cos(rad)) / IMG_W
                    k2y = (row['center_y'] - LEVER_ARM * np.sin(rad)) / IMG_H
                    
                    # YOLO-Pose Format: cls cx cy w h k1x k1y v1 k2x k2y v2
                    f.write(f"0 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f} {k1x:.6f} {k1y:.6f} 2 {k2x:.6f} {k2y:.6f} 2\n")

    # Create YAML
    yaml_path = os.path.join(fold_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {os.path.abspath(fold_dir)}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write(f"kpt_shape: [2, 3]\n")
        f.write(f"names: {{0: 'tube'}}\n")

print(f"Pose dataset prepared in {BASE_DIR}")
