"""
Collage Montage Generator - Refined SOTA Edition
Generates a large grid GIF where multiple random images simultaneously
progress through the detection pipeline stages:
  Stage 1: Raw Image
  Stage 2: YOLO11s-Pose Detection (orange boxes)
  Stage 3: SAM Segmentation (cyan masks)
  Stage 4: PCA Axis (white lines)
  Stage 5: Final Refined Orientation (green lines)
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.math_utils import angle_diff
from utils.eval_utils import get_pca_angle

import cv2
import pandas as pd
import numpy as np
import math
import os
import glob
import random
import torch
from ultralytics import YOLO, SAM
import imageio

# ── Configuration ──────────────────────────────────────────────────────────
NUM_IMAGES = 8            # total images in the collage
GRID_COLS = 4             # columns in the grid
GRID_ROWS = 2             # rows in the grid
STAGE_FRAMES = 8          # frames to hold each stage
FPS = 12                  
SEED = 42                 
RESIZE_W = 320            
RESIZE_H = 240            
BANNER_H = 60             

random.seed(SEED)
os.makedirs('outputs', exist_ok=True)

# ── Load Models ────────────────────────────────────────────────────────────
print("Loading Refined SOTA Models...")

# Map images to folds
img_to_fold = {}
for fold in range(5):
    val_dir = f'datasets_obb/fold_{fold}/images/val'
    if os.path.exists(val_dir):
        for img in os.listdir(val_dir):
            img_to_fold[img] = fold

yolo_models = {}
for fold in range(5):
    path = f'models/runs_pose/fold_{fold}.pt'
    if os.path.exists(path):
        yolo_models[fold] = YOLO(path)

sam_model = SAM('models/mobile_sam.pt')
print(f"Loaded {len(yolo_models)} YOLO11s-Pose models and MobileSAM.")

# ── Select random images ───────────────────────────────────────────────────
all_images = [f for f in os.listdir('images') if f.endswith('-color.png')]
selected_images = random.sample(all_images, min(NUM_IMAGES, len(all_images)))
print(f"Selected images: {selected_images}")

# ── Process each image ─────────────────────────────────────────────────────
def process_image(img_name):
    img_path = f'images/{img_name}'
    fold = img_to_fold.get(img_name, 0)
    if fold not in yolo_models: fold = 0
    yolo_model = yolo_models[fold]

    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    stages = []

    # Stage 1: Raw
    frame = img_rgb.copy()
    stages.append(cv2.resize(frame, (RESIZE_W, RESIZE_H)))

    # Stage 2: YOLO Detection
    results = yolo_model(img_bgr, verbose=False)[0]
    yolo_frame = img_rgb.copy()
    boxes = []
    kpts_list = []
    if results.boxes is not None:
        for i, box in enumerate(results.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = box
            cv2.rectangle(yolo_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 165, 0), 2)
            boxes.append(box.astype(int))
            kpts_list.append(results.keypoints.xy.cpu().numpy()[i])
    stages.append(cv2.resize(yolo_frame, (RESIZE_W, RESIZE_H)))

    # Stage 3: SAM
    sam_frame = img_rgb.copy()
    overlay = np.zeros_like(img_rgb)
    tube_data = []

    for i, bbox in enumerate(boxes):
        sam_res = sam_model.predict(img_bgr, bboxes=[bbox], verbose=False)[0]
        if len(sam_res.masks) > 0:
            mask = sam_res.masks.data[0].cpu().numpy().astype(np.uint8)
            mask = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            color_mask = np.zeros_like(img_rgb)
            color_mask[mask > 0] = [0, 255, 255]
            overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.5, 0)
            
            cx, cy, pca_axis = get_pca_angle(mask)
            if pca_axis is not None:
                # Pose Direction Logic
                p_center, p_tip = kpts_list[i][0], kpts_list[i][1]
                pose_dir = (math.degrees(math.atan2(p_center[1] - p_tip[1], p_tip[0] - p_center[0])) + 360) % 360
                
                if angle_diff(pca_axis, pose_dir) > 90:
                    final = (pca_axis + 180) % 360
                else:
                    final = pca_axis
                tube_data.append((cx, cy, pca_axis, final))

    sam_frame = cv2.addWeighted(sam_frame, 1.0, overlay, 0.4, 0)
    stages.append(cv2.resize(sam_frame, (RESIZE_W, RESIZE_H)))

    # Stage 4: PCA
    pca_frame = sam_frame.copy()
    for cx, cy, pca_ang, _ in tube_data:
        rad = math.radians(pca_ang)
        p1 = (int(cx + 25 * math.cos(rad)), int(cy - 25 * math.sin(rad)))
        p2 = (int(cx - 25 * math.cos(rad)), int(cy + 25 * math.sin(rad)))
        cv2.line(pca_frame, p1, p2, (255, 255, 255), 2, cv2.LINE_AA)
    stages.append(cv2.resize(pca_frame, (RESIZE_W, RESIZE_H)))

    # Stage 5: Final
    final_frame = img_rgb.copy()
    for cx, cy, _, final_ang in tube_data:
        rad = math.radians(final_ang)
        tx = int(cx + 35 * math.cos(rad))
        ty = int(cy - 35 * math.sin(rad))
        cv2.circle(final_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
        cv2.line(final_frame, (int(cx), int(cy)), (tx, ty), (0, 255, 0), 3, cv2.LINE_AA)
    stages.append(cv2.resize(final_frame, (RESIZE_W, RESIZE_H)))

    return stages

# ── Build frames ──────────────────────────────────────────────────────────
print("Generating collage frames...")
image_stages = [process_image(name) for name in selected_images]

num_stages = 5
actual_rows = 2
grid_h, grid_w = actual_rows * RESIZE_H, GRID_COLS * RESIZE_W
frame_h, frame_w = BANNER_H + grid_h, grid_w

stage_labels = ["RAW IMAGE", "YOLO11s-POSE", "REFINED SEGMENTATION", "PCA AXIS", "FINAL REFINED"]
stage_colors = [(200, 200, 200), (255, 165, 0), (0, 255, 255), (255, 255, 255), (0, 255, 0)]

all_frames = []
for s_idx in range(num_stages):
    for _ in range(STAGE_FRAMES):
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        for i, stages in enumerate(image_stages):
            r, c = i // GRID_COLS, i % GRID_COLS
            grid[r*RESIZE_H:(r+1)*RESIZE_H, c*RESIZE_W:(c+1)*RESIZE_W] = stages[s_idx]
            cv2.rectangle(grid, (c*RESIZE_W, r*RESIZE_H), ((c+1)*RESIZE_W, (r+1)*RESIZE_H), (60, 60, 60), 1)

        full_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        full_frame[BANNER_H:, :] = grid
        banner = np.full((BANNER_H, frame_w, 3), (30, 30, 40), dtype=np.uint8)
        full_frame[:BANNER_H, :] = banner
        
        lbl = stage_labels[s_idx]
        (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        cv2.putText(full_frame, lbl, ((frame_w-tw)//2, (BANNER_H+th)//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, stage_colors[s_idx], 3)
        cv2.line(full_frame, (0, BANNER_H), (frame_w, BANNER_H), stage_colors[s_idx], 2)
        all_frames.append(full_frame)

imageio.mimsave('outputs/collage_montage.gif', all_frames, fps=FPS)
cv2.imwrite('outputs/final_predictions.png', cv2.cvtColor(all_frames[-1][BANNER_H:, :], cv2.COLOR_RGB2BGR))
print("Saved outputs/collage_montage.gif and outputs/final_predictions.png")
