"""Prepare OBB Datasets for 5-Fold Cross-Validation.

Converts CSV annotations into YOLO OBB format (4-corner polygon) and splits
images into 5 train/val folds.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import shutil
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold

df = pd.read_csv('annotations.csv')
img_w, img_h = 640, 480
images = df['image'].unique()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Bounding box dimensions (pixels) for the OBB polygon.
# W=40, H=26 approximate the elongated shape of the tube lid + tab.
# The 5px X-offset shifts the box center toward the tab to improve
# downstream YOLO learning of the asymmetric orientation.
OBB_WIDTH = 40
OBB_HEIGHT = 26
OBB_X_OFFSET = 5

def get_oriented_corners(cx, cy, angle_deg):
    """Generate 4 rotated corners of an OBB centered at (cx, cy)."""
    W, H = OBB_WIDTH, OBB_HEIGHT
    corners = np.array([
        [-W/2 + OBB_X_OFFSET, -H/2],
        [ W/2 + OBB_X_OFFSET, -H/2],
        [ W/2 + OBB_X_OFFSET,  H/2],
        [-W/2 + OBB_X_OFFSET,  H/2],
    ])
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)

    rotated_corners = []
    for pt in corners:
        x, y = pt[0], pt[1]
        rx = x * cos_a + y * sin_a
        ry = -x * sin_a + y * cos_a
        rotated_corners.append([rx + cx, ry + cy])
    return rotated_corners

def prepare_yolo_obb_format(df_subset, images_subset, fold, split_name):
    """Write images and YOLO OBB label files for one fold/split."""
    img_dir = f'datasets_obb/fold_{fold}/images/{split_name}'
    lbl_dir = f'datasets_obb/fold_{fold}/labels/{split_name}'
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for img_name in images_subset:
        src_img = f'images/{img_name}'
        dst_img = f'{img_dir}/{img_name}'
        shutil.copy(src_img, dst_img)

        img_df = df_subset[df_subset['image'] == img_name]
        lbl_file = f'{lbl_dir}/{img_name.replace(".png", ".txt")}'

        with open(lbl_file, 'w') as f:
            for _, row in img_df.iterrows():
                cx, cy = row['center_x'], row['center_y']
                angle_deg = row['angle_deg']
                corners = get_oriented_corners(cx, cy, angle_deg)

                # Normalize corners to [0, 1]
                norm_corners = []
                for pt in corners:
                    nx = pt[0] / img_w
                    ny = pt[1] / img_h
                    norm_corners.extend([nx, ny])

                line = f"0 {' '.join([f'{c:.6f}' for c in norm_corners])}\n"
                f.write(line)

for fold, (train_idx, val_idx) in enumerate(kf.split(images)):
    train_imgs = images[train_idx]
    val_imgs = images[val_idx]

    prepare_yolo_obb_format(df, train_imgs, fold, 'train')
    prepare_yolo_obb_format(df, val_imgs, fold, 'val')

    yaml_content = f"""
path: {os.path.abspath(f'datasets_obb/fold_{fold}')}
train: images/train
val: images/val

names:
  0: tube
"""
    with open(f'datasets_obb/fold_{fold}/dataset.yaml', 'w') as f:
        f.write(yaml_content)

print("Created 5 folds in datasets_obb/")
