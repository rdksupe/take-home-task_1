"""Evaluate the full pipeline: YOLOv8 OBB -> MobileSAM -> PCA -> ResNet-18 SVM.

Runs 5-fold cross-validation where every tube is evaluated exactly once.
Reports detection metrics (precision, recall, F1), angle error statistics,
and per-fold breakdowns.

Note: YOLOv8 OBB is used instead of YOLOv26 OBB because it achieves 100%
recall (371/371 tubes detected), whereas YOLOv26 drops 4 tubes under extreme
lighting. The trade-off is a slightly higher MAE (~4.66° vs 4.41°).
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.math_utils import angle_diff
from utils.eval_utils import evaluate_single_image, TP_MATCH_THRESHOLD_PX

import math
import numpy as np
import pandas as pd
import cv2
import glob
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO, SAM
import pickle

df = pd.read_csv('annotations.csv')
sam_model = SAM('models/mobile_sam.pt')

all_tps = 0
all_fps = 0
all_gt_count = 0
all_skipped = 0
angle_errors = []
fold_results = []

for fold in range(5):
    model_paths = glob.glob(f'models/runs_yolov8/fold_{fold}*/weights/best.pt')
    if not model_paths:
        print(f"Missing model for fold {fold}")
        continue

    model_paths.sort(key=os.path.getmtime)
    model_path = model_paths[-1]

    svm_path = f'models/resnet_svm_fold_{fold}.pkl'
    if not os.path.exists(svm_path):
        print(f"Missing SVM for fold {fold}")
        continue

    with open(svm_path, 'rb') as f:
        svm_clf = pickle.load(f)

    obb_model = YOLO(model_path)
    val_dir = f'datasets_obb/fold_{fold}/images/val'
    val_images = os.listdir(val_dir)

    fold_tps = 0
    fold_fps = 0
    fold_gt = 0
    fold_errors = []

    for img_name in val_images:
        gt_subset = df[df['image'] == img_name]
        gt_centers = np.array([[row['center_x'], row['center_y']] for _, row in gt_subset.iterrows()])
        gt_angles = [row['angle_deg'] for _, row in gt_subset.iterrows()]
        fold_gt += len(gt_centers)

        img_path = os.path.join(val_dir, img_name)
        img = cv2.imread(img_path)

        pred_centers, pred_angles = evaluate_single_image(img, obb_model, sam_model, svm_clf)

        if not pred_centers:
            all_skipped += len(gt_centers)
            continue
        pred_centers = np.array(pred_centers)

        dist_mat = np.zeros((len(gt_centers), len(pred_centers)))
        for i, gt_c in enumerate(gt_centers):
            for j, p_c in enumerate(pred_centers):
                dist_mat[i, j] = np.linalg.norm(gt_c - p_c)

        row_ind, col_ind = linear_sum_assignment(dist_mat)
        matched_preds = set()
        for r, c in zip(row_ind, col_ind):
            if dist_mat[r, c] < TP_MATCH_THRESHOLD_PX:
                fold_tps += 1
                matched_preds.add(c)
                final_angle = pred_angles[c]
                err = angle_diff(gt_angles[r], final_angle)
                fold_errors.append(err)

        # Count unmatched predictions as false positives
        fold_fps += len(pred_centers) - len(matched_preds)

    fold_results.append({
        'fold': fold,
        'tps': fold_tps,
        'fps': fold_fps,
        'gt': fold_gt,
        'errors': fold_errors,
    })
    all_tps += fold_tps
    all_fps += fold_fps
    all_gt_count += fold_gt
    angle_errors.extend(fold_errors)

# --- Per-Fold Breakdown ---
print("\n--- PER-FOLD RESULTS ---")
print(f"{'Fold':<6} {'TPs':<6} {'FPs':<6} {'GT':<6} {'Recall':<10} {'MAE':<10} {'Median':<10} {'<10°':<10}")
for fr in fold_results:
    errs = np.array(fr['errors'])
    recall = fr['tps'] / fr['gt'] if fr['gt'] > 0 else 0
    mae = np.mean(errs) if len(errs) > 0 else float('nan')
    med = np.median(errs) if len(errs) > 0 else float('nan')
    lt10 = (np.sum(errs < 10) / len(errs) * 100) if len(errs) > 0 else 0
    print(f"{fr['fold']:<6} {fr['tps']:<6} {fr['fps']:<6} {fr['gt']:<6} {recall:<10.4f} {mae:<10.2f} {med:<10.2f} {lt10:<10.2f}%")

# --- Aggregate Results ---
errors = np.array(angle_errors)
os.makedirs('outputs', exist_ok=True)
np.save('outputs/baseline_errors.npy', errors)

mean_err = np.mean(errors)
median_err = np.median(errors)

precision = all_tps / (all_tps + all_fps) if (all_tps + all_fps) > 0 else 0
recall = all_tps / all_gt_count if all_gt_count > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

def pct_less_than(threshold):
    return (np.sum(errors < threshold) / len(errors)) * 100

def pct_between(low, high):
    return (np.sum((errors >= low) & (errors < high)) / len(errors)) * 100

print(f"\n{'='*55}")
print(f"  AGGREGATE RESULTS (5-Fold CV, {all_gt_count} GT tubes)")
print(f"{'='*55}")
print(f"  Matched TPs:    {all_tps}")
print(f"  False Positives: {all_fps}")
print(f"  Undetected:      {all_gt_count - all_tps}")
print(f"  Precision:       {precision:.4f}")
print(f"  Recall:          {recall:.4f}  ({all_tps}/{all_gt_count})")
print(f"  F1 Score:        {f1:.4f}")
print(f"")
print(f"  Mean Angle Error:   {mean_err:.2f}°")
print(f"  Median Angle Error: {median_err:.2f}°")
print(f"")
print(f"  Errors 0° - 1°:  {pct_between(0, 1):.2f}%")
print(f"  Errors 1° - 2°:  {pct_between(1, 2):.2f}%")
print(f"  Errors 2° - 3°:  {pct_between(2, 3):.2f}%")
print(f"  Errors 3° - 4°:  {pct_between(3, 4):.2f}%")
print(f"  Errors 4° - 5°:  {pct_between(4, 5):.2f}%")
print(f"  Errors < 5°:     {pct_less_than(5):.2f}%")
print(f"  Errors < 10°:    {pct_less_than(10):.2f}%")
print(f"  Errors < 22°:    {pct_less_than(22):.2f}%")
print(f"  Flip Failures (>= 90°): {(np.sum(errors >= 90) / len(errors)) * 100:.2f}%")
