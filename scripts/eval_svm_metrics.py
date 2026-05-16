"""Evaluate the ResNet-18 SVM patch classifier in isolation.

Reports per-fold and aggregate classification metrics (accuracy, precision,
recall, F1) for the tab/joint binary classifier.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.image_utils import get_patches_from_gt
from utils.model_utils import extract_features

import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle



df = pd.read_csv('annotations.csv')

# ======================================================================
#  PART 1: SVM Classifier Metrics (unchanged)
# ======================================================================
all_y_true_train = []
all_y_pred_train = []
all_y_true_val = []
all_y_pred_val = []

for fold in range(5):
    train_dir = f'datasets_obb/fold_{fold}/images/train'
    val_dir = f'datasets_obb/fold_{fold}/images/val'

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        continue

    # --- TRAINING DATA EXTRACTION ---
    train_images = os.listdir(train_dir)
    X_train, y_train = [], []

    for img_name in train_images:
        img_path = os.path.join(train_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_subset = df[df['image'] == img_name]
        for _, row in gt_subset.iterrows():
            cx, cy = row['center_x'], row['center_y']
            ang = row['angle_deg']

            tab_p, joint_p = get_patches_from_gt(cx, cy, ang, img)

            if tab_p is not None:
                X_train.append(extract_features(tab_p))
                y_train.append(1)  # Tab
            if joint_p is not None:
                X_train.append(extract_features(joint_p))
                y_train.append(0)  # Joint

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Load the already trained SVM for this fold
    svm_path = f'models/resnet_svm_fold_{fold}.pkl'
    with open(svm_path, 'rb') as f:
        clf = pickle.load(f)

    # Training Metrics
    y_pred_train = clf.predict(X_train)
    all_y_true_train.extend(y_train)
    all_y_pred_train.extend(y_pred_train)

    # --- VALIDATION DATA EXTRACTION ---
    val_images = os.listdir(val_dir)
    X_val, y_val = [], []

    for img_name in val_images:
        img_path = os.path.join(val_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        gt_subset = df[df['image'] == img_name]
        for _, row in gt_subset.iterrows():
            cx, cy = row['center_x'], row['center_y']
            ang = row['angle_deg']

            tab_p, joint_p = get_patches_from_gt(cx, cy, ang, img)

            if tab_p is not None:
                X_val.append(extract_features(tab_p))
                y_val.append(1)  # Tab
            if joint_p is not None:
                X_val.append(extract_features(joint_p))
                y_val.append(0)  # Joint

    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Validation Metrics
    y_pred_val = clf.predict(X_val)
    all_y_true_val.extend(y_val)
    all_y_pred_val.extend(y_pred_val)

# Calculate aggregate metrics across all 5 folds
print("========================================")
print("  ResNet-18 SVM Classifier Metrics")
print("  (Tab = Class 1 | Joint = Class 0)")
print("========================================")

print("\n--- TRAINING METRICS ---")
print(f"Accuracy:  {accuracy_score(all_y_true_train, all_y_pred_train):.4f}")
print(f"Precision: {precision_score(all_y_true_train, all_y_pred_train):.4f}")
print(f"Recall:    {recall_score(all_y_true_train, all_y_pred_train):.4f}")
print(f"F1-Score:  {f1_score(all_y_true_train, all_y_pred_train):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(all_y_true_train, all_y_pred_train))

print("\n--- VALIDATION (TEST) METRICS ---")
print(f"Accuracy:  {accuracy_score(all_y_true_val, all_y_pred_val):.4f}")
print(f"Precision: {precision_score(all_y_true_val, all_y_pred_val):.4f}")
print(f"Recall:    {recall_score(all_y_true_val, all_y_pred_val):.4f}")
print(f"F1-Score:  {f1_score(all_y_true_val, all_y_pred_val):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(all_y_true_val, all_y_pred_val))



