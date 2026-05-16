"""
Collage Montage Generator
Generates a large grid GIF where multiple random images simultaneously
progress through the detection pipeline stages:
  Stage 1: Raw Image
  Stage 2: YOLO Detection (orange boxes)
  Stage 3: SAM Segmentation (cyan masks)
  Stage 4: PCA Axis (white lines)
  Stage 5: Final Orientation (red lines)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.math_utils import angle_diff
from utils.image_utils import extract_patch, get_patches_from_gt
from utils.model_utils import extract_features, resolve_180_resnet, get_resnet_model

import cv2
import pandas as pd
import numpy as np
import math
import os
import glob
import random
import torch
from ultralytics import YOLO, SAM
import pickle
import imageio

# ── Configuration ──────────────────────────────────────────────────────────
NUM_IMAGES = 8            # total images in the collage
GRID_COLS = 4             # columns in the grid (rows = ceil(NUM_IMAGES / GRID_COLS))
GRID_ROWS = 2             # rows in the grid
STAGE_FRAMES = 6          # how many frames to hold each stage before advancing
FPS = 10                  # frames per second
SEED = 42                 # random seed for reproducibility (set to None for true random)
RESIZE_W = 320            # resize each cell to this width
RESIZE_H = 240            # resize each cell to this height
BANNER_H = 60             # height of the top banner showing the current stage name

random.seed(SEED)

os.makedirs('outputs', exist_ok=True)

# ── Load Models ────────────────────────────────────────────────────────────
print("Loading models...")

# Map images to their validation fold so we use the correct model
img_to_fold = {}
for fold in range(5):
    val_dir = f'datasets/fold_{fold}/images/val'
    if os.path.exists(val_dir):
        for img in os.listdir(val_dir):
            img_to_fold[img] = fold
    val_dir_obb = f'datasets_obb/fold_{fold}/images/val'
    if os.path.exists(val_dir_obb):
        for img in os.listdir(val_dir_obb):
            img_to_fold[img] = fold

# Load all 5 YOLO models + SVM models
# Try runs_yolo26 first (full repo), then falls back to runs_yolov8 (clean/take-home)
yolo_models = {}
svm_models = {}
for fold in range(5):
    model_paths = glob.glob(f'models/runs_yolo26/fold_{fold}*/weights/best.pt')
    if not model_paths:
        model_paths = glob.glob(f'models/runs_yolov8/fold_{fold}*/weights/best.pt')
    if not model_paths:
        continue
    model_paths.sort(key=os.path.getmtime)
    yolo_models[fold] = YOLO(model_paths[-1])
    svm_path = f'models/resnet_svm_fold_{fold}.pkl'
    if os.path.exists(svm_path):
        with open(svm_path, 'rb') as f:
            svm_models[fold] = pickle.load(f)

sam_model = SAM('models/mobile_sam.pt')
print(f"Loaded {len(yolo_models)} YOLO models, {len(svm_models)} SVM models, MobileSAM")

# ── Select random images ───────────────────────────────────────────────────
all_images = [f for f in os.listdir('images') if f.endswith('-color.png')]
selected_images = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
print(f"Selected images: {selected_images}")

# ── Process each image through all stages ──────────────────────────────────
# Each image produces a list of 5 stage frames (each already resized to cell size)
def process_image(img_name):
    """Process a single image through all 5 pipeline stages, return list of resized frames."""
    img_path = f'images/{img_name}'
    if not os.path.exists(img_path):
        return None

    fold = img_to_fold.get(img_name, 0)
    if fold not in yolo_models:
        fold = list(yolo_models.keys())[0]

    yolo_model = yolo_models[fold]
    svm_clf = svm_models[fold]

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    stages = []

    # Stage 1: Raw Image
    frame = img_rgb.copy()
    cv2.putText(frame, "Raw", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    stages.append(cv2.resize(frame, (RESIZE_W, RESIZE_H)))

    # Stage 2: YOLO Detection
    results = yolo_model(img_bgr, verbose=False)
    yolo_frame = img_rgb.copy()
    cv2.putText(yolo_frame, "YOLO", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)

    boxes = []
    if results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = box
            cv2.rectangle(yolo_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 165, 0), 2)
            boxes.append([int(x1 - 5), int(y1 - 5), int(x2 + 5), int(y2 + 5)])
    stages.append(cv2.resize(yolo_frame, (RESIZE_W, RESIZE_H)))

    # Stage 3: SAM Segmentation
    sam_frame = img_rgb.copy()
    cv2.putText(sam_frame, "SAM", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    tube_data = []
    overlay = np.zeros_like(img_rgb)

    if boxes:
        for bbox in boxes:
            sam_res = sam_model.predict(img_bgr, bboxes=[bbox], verbose=False)
            if len(sam_res[0].masks) > 0:
                mask = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8)
                mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)

                color_mask = np.zeros_like(img_rgb)
                color_mask[mask > 0] = [0, 200, 255]
                overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)

                mask_u8 = mask * 255
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    cv2.drawContours(sam_frame, [c], -1, (0, 255, 255), 2)
                    M = cv2.moments(c)
                    if M['m00'] != 0:
                        cx, cy = M['m10'] / M['m00'], M['m01'] / M['m00']
                        theta = 0.5 * math.atan2(2 * M['mu11'], M['mu20'] - M['mu02'])
                        pca_ang = (-math.degrees(theta) + 360) % 180
                        tube_data.append((cx, cy, pca_ang))

    sam_frame = cv2.addWeighted(sam_frame, 1.0, overlay, 0.5, 0)
    stages.append(cv2.resize(sam_frame, (RESIZE_W, RESIZE_H)))

    # Stage 4: PCA Axis
    pca_frame = sam_frame.copy()
    cv2.putText(pca_frame, "PCA", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for cx, cy, pca_ang in tube_data:
        rad = math.radians(pca_ang)
        p1 = (int(cx + 25 * math.cos(rad)), int(cy - 25 * math.sin(rad)))
        p2 = (int(cx - 25 * math.cos(rad)), int(cy + 25 * math.sin(rad)))
        cv2.line(pca_frame, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
    stages.append(cv2.resize(pca_frame, (RESIZE_W, RESIZE_H)))

    # Stage 5: Final Orientation
    final_frame = img_rgb.copy()
    cv2.putText(final_frame, "Final", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    for cx, cy, pca_ang in tube_data:
        final_ang = resolve_180_resnet(cx, cy, pca_ang, img_bgr, svm_clf)
        rad = math.radians(final_ang)
        tx = int(cx + 30 * math.cos(rad))
        ty = int(cy - 30 * math.sin(rad))
        cv2.circle(final_frame, (int(cx), int(cy)), 5, (255, 0, 0), -1)
        cv2.line(final_frame, (int(cx), int(cy)), (tx, ty), (255, 0, 0), 3, cv2.LINE_AA)
    stages.append(cv2.resize(final_frame, (RESIZE_W, RESIZE_H)))

    return stages


# ── Build grid frames ──────────────────────────────────────────────────────
print("Processing images...")
image_stages = []
for img_name in selected_images:
    stages = process_image(img_name)
    if stages:
        image_stages.append(stages)
    print(f"  Processed {img_name}")

num_stages = 5  # Raw, YOLO, SAM, PCA, Final
total_images = len(image_stages)
actual_rows = (total_images + GRID_COLS - 1) // GRID_COLS

# Grid area is below the banner
grid_h = actual_rows * RESIZE_H
grid_w = GRID_COLS * RESIZE_W
banner_h = BANNER_H
frame_h = banner_h + grid_h
frame_w = grid_w

all_frames = []
stage_labels = ["RAW IMAGE", "YOLO DETECTION", "SAM SEGMENTATION", "PCA AXIS", "FINAL ORIENTATION"]
stage_colors = [
    (200, 200, 200),
    (255, 165, 0),
    (0, 255, 255),
    (255, 255, 255),
    (255, 0, 0),
]

for stage_idx in range(num_stages):
    for repeat in range(STAGE_FRAMES):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        grid[:] = [30, 30, 30]  # dark gray background

        for img_idx, stages in enumerate(image_stages):
            row = img_idx // GRID_COLS
            col = img_idx % GRID_COLS
            y_start = row * RESIZE_H
            x_start = col * RESIZE_W

            cell_frame = stages[stage_idx]
            grid[y_start:y_start + RESIZE_H, x_start:x_start + RESIZE_W] = cell_frame

            # Draw border
            cv2.rectangle(grid, (x_start, y_start),
                          (x_start + RESIZE_W, y_start + RESIZE_H),
                          (100, 100, 100), 1)

        # Compose full frame: banner on top + grid below
        full_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        full_frame[banner_h:banner_h + grid_h, :] = grid

        # Draw banner with stage label
        banner_bg = np.zeros((banner_h, frame_w, 3), dtype=np.uint8)
        banner_color = stage_colors[stage_idx]
        banner_bg[:] = [max(c - 180, 0) for c in banner_color]  # dark tint of the stage color
        banner_bg[:] = [40, 40, 50]  # dark blue-gray
        full_frame[:banner_h, :] = banner_bg

        # Stage label centered in banner
        label = stage_labels[stage_idx]
        font_scale = 1.8
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
        tx = (frame_w - tw) // 2
        ty = (banner_h + th) // 2 + 5
        cv2.putText(full_frame, label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, stage_colors[stage_idx], 3)

        # Draw thin line under banner in the stage color
        cv2.line(full_frame, (0, banner_h), (frame_w, banner_h), stage_colors[stage_idx], 2)

        all_frames.append(full_frame)

    # Transition: overlay the next stage label over the current grid (fade-style)
    if stage_idx < num_stages - 1:
        next_label = stage_labels[stage_idx + 1]
        next_color = stage_colors[stage_idx + 1]
        for _ in range(3):
            # Rebuild the current grid
            trans_grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
            trans_grid[:] = [30, 30, 30]
            for img_idx, stages in enumerate(image_stages):
                row = img_idx // GRID_COLS
                col = img_idx % GRID_COLS
                y_start = row * RESIZE_H
                x_start = col * RESIZE_W
                cell_frame = stages[stage_idx]
                trans_grid[y_start:y_start + RESIZE_H, x_start:x_start + RESIZE_W] = cell_frame
                cv2.rectangle(trans_grid, (x_start, y_start),
                              (x_start + RESIZE_W, y_start + RESIZE_H),
                              (100, 100, 100), 1)
            full_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
            full_frame[banner_h:banner_h + grid_h, :] = trans_grid
            full_frame[:banner_h, :] = [20, 20, 30]
            (tw, th), _ = cv2.getTextSize(next_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 3)
            tx = (frame_w - tw) // 2
            ty = (banner_h + th) // 2 + 5
            cv2.putText(full_frame, f"Next: {next_label}", (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, next_color, 3)
            cv2.line(full_frame, (0, banner_h), (frame_w, banner_h), next_color, 2)
            all_frames.append(full_frame)

# ── Save GIF ───────────────────────────────────────────────────────────────
output_path = 'outputs/collage_montage.gif'
print(f"Generating collage GIF ({len(all_frames)} frames @ {FPS} fps)...")
imageio.mimsave(output_path, all_frames, fps=FPS)
print(f"Saved {output_path}")

# Also save the final stage as a PNG (use the last actual frame — the final orientation stage)
print("Saving final collage PNG...")
final_frame = all_frames[-1]  # last frame = final orientation stage
cv2.imwrite('outputs/collage_final.png', cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))
print("Saved outputs/collage_final.png")
