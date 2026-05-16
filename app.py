"""Streamlit dashboard for visualizing pipeline predictions vs ground truth."""
from utils.math_utils import angle_diff
from utils.image_utils import extract_patch, get_patches_from_gt
from utils.model_utils import extract_features, resolve_180_resnet, get_resnet_model

import streamlit as st
import cv2
import pandas as pd
import numpy as np
import math
import os
import torch
import torchvision.transforms as T
import torchvision.models as models
from ultralytics import YOLO, SAM
import pickle
import glob

st.set_page_config(layout="wide")
st.title("Tube Detection & Orientation Pipeline")
st.markdown("### YOLOv8 OBB → MobileSAM → OpenCV Moments → ResNet-18 SVM")

@st.cache_data
def load_annotations():
    return pd.read_csv('annotations.csv')

@st.cache_resource
def load_sam():
    return SAM('models/mobile_sam.pt')

@st.cache_resource
def load_models():
    yolo_models = {}
    svm_models = {}
    for fold in range(5):
        model_paths = glob.glob(f'models/runs_yolov8/fold_{fold}*/weights/best.pt')
        if not model_paths:
            continue
        model_paths.sort(key=os.path.getmtime)
        yolo_path = model_paths[-1]
        svm_path = f'models/resnet_svm_fold_{fold}.pkl'

        if os.path.exists(yolo_path) and os.path.exists(svm_path):
            yolo_models[fold] = YOLO(yolo_path)
            with open(svm_path, 'rb') as f:
                svm_models[fold] = pickle.load(f)
    return yolo_models, svm_models

@st.cache_data
def get_image_folds():
    img_to_fold = {}
    for fold in range(5):
        val_dir = f'datasets_obb/fold_{fold}/images/val'
        if os.path.exists(val_dir):
            for img in os.listdir(val_dir):
                img_to_fold[img] = fold
    return img_to_fold

df = load_annotations()
sam_model = load_sam()
yolo_models, svm_models = load_models()
img_to_fold = get_image_folds()

images = sorted(list(img_to_fold.keys()))
if not images:
    st.error("No images found in validation folds. Run reproduce.sh first.")
    st.stop()

selected_img = st.selectbox("Select an image:", images)
fold = img_to_fold[selected_img]

st.write(f"**Validation Fold:** {fold}")

# Load image
img_path = f'images/{selected_img}'
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    st.error(f"Could not load image: {img_path}")
    st.stop()
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Ground Truth
gt_subset = df[df['image'] == selected_img]

yolo_model = yolo_models.get(fold)
svm_model_fold = svm_models.get(fold)

pred_data = []
img = cv2.imread(img_path)

if yolo_model:
    results = yolo_model(img_path, verbose=False)
    obb = results[0].obb
    if obb is not None and len(obb) > 0:
        # OBB gives 4-corner polygons; convert to axis-aligned bboxes for SAM
        corners = obb.xyxyxyxy.cpu().numpy()  # shape (N, 4, 2)
        for pts in corners:
            x1, y1 = pts[:, 0].min(), pts[:, 1].min()
            x2, y2 = pts[:, 0].max(), pts[:, 1].max()
            box = [x1, y1, x2, y2]
            bbox = [int(x1-5), int(y1-5), int(x2+5), int(y2+5)]
            sam_res = sam_model.predict(img, bboxes=[bbox], verbose=False)
            if len(sam_res[0].masks) > 0:
                mask = sam_res[0].masks.data[0].cpu().numpy().astype(np.uint8) * 255
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    M = cv2.moments(c)
                    if M['m00'] != 0:
                        cx, cy = M['m10']/M['m00'], M['m01']/M['m00']
                        theta = 0.5 * math.atan2(2*M['mu11'], M['mu20']-M['mu02'])
                        pca_ang = (-math.degrees(theta) + 360) % 180
                        final_ang = resolve_180_resnet(cx, cy, pca_ang, img, svm_model_fold)
                        pred_data.append({
                            'pred_center_x': cx,
                            'pred_center_y': cy,
                            'pred_angle_deg': final_ang
                        })


def draw_visualization(image, gt, preds_list):
    vis_img = image.copy()
    R = 30

    # Draw GT in Green
    for _, row in gt.iterrows():
        cx_gt, cy_gt = int(row['center_x']), int(row['center_y'])
        angle_deg = row['angle_deg']
        rad = math.radians(angle_deg)
        tx = int(cx_gt + R * math.cos(rad))
        ty = int(cy_gt - R * math.sin(rad))
        cv2.circle(vis_img, (cx_gt, cy_gt), 4, (0, 255, 0), -1)
        cv2.line(vis_img, (cx_gt, cy_gt), (tx, ty), (0, 255, 0), 2)

    # Draw Preds in Red
    for pred in preds_list:
        cx_p, cy_p = int(pred['pred_center_x']), int(pred['pred_center_y'])
        final_angle = pred['pred_angle_deg']
        rad = math.radians(final_angle)
        tx = int(cx_p + R * math.cos(rad))
        ty = int(cy_p - R * math.sin(rad))
        cv2.circle(vis_img, (cx_p, cy_p), 4, (255, 0, 0), -1)
        cv2.line(vis_img, (cx_p, cy_p), (tx, ty), (255, 0, 0), 2)

    return vis_img

vis_img = draw_visualization(img_rgb, gt_subset, pred_data)

col1, col2 = st.columns(2)
with col1:
    st.subheader("Pipeline Visualization")
    st.image(vis_img, use_column_width=True, caption="Green: Ground Truth | Red: Pipeline Prediction")

with col2:
    st.subheader("Data Comparison")
    st.write("**Ground Truth:**")
    st.dataframe(gt_subset[['center_x', 'center_y', 'angle_deg']])

    st.write("**Pipeline Predictions:**")
    if pred_data:
        st.dataframe(pd.DataFrame(pred_data))
    else:
        st.write("No predictions.")
