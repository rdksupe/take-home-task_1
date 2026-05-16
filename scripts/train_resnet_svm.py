"""Train ResNet-18 + SVM patch classifiers for 180-degree symmetry resolution.

For each fold, extracts tab/joint patches from training images using ground-truth
annotations, computes ResNet-18 feature embeddings, and trains an RBF-SVM.
Saves one classifier per fold to models/resnet_svm_fold_{N}.pkl.

Note: The SVM is trained on patches extracted at ground-truth centers/angles.
At inference, patches are extracted from PCA-estimated centers/angles, which
introduces a small distribution shift. This is acceptable because the PCA
estimates are typically within 2-3 degrees of ground truth.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.image_utils import get_patches_from_gt
from utils.model_utils import extract_features

import cv2
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pickle

df = pd.read_csv('annotations.csv')
os.makedirs('models', exist_ok=True)

for fold in range(5):
    print(f"Extracting ResNet features and training SVM for Fold {fold}...")
    train_dir = f'datasets_obb/fold_{fold}/images/train'
    if not os.path.exists(train_dir):
        continue

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

    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', gamma='auto', probability=True))
    clf.fit(X_train, y_train)

    svm_path = f'models/resnet_svm_fold_{fold}.pkl'
    with open(svm_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"  Saved {svm_path} (trained on {len(X_train)} patches)")

print("ResNet+SVM training completed for all folds.")
