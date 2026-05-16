#!/bin/bash
# ===========================================
#  Tube Detection & Orientation Pipeline
#  Full Reproduction Script
# ===========================================
#
# Prerequisites:
#   pip install -r requirements.txt
#
# Note: Step 2 trains 5 YOLO models for 100 epochs each.
# A dedicated CUDA GPU is highly recommended.

set -e  # Exit on error

echo "==========================================="
echo " Tube Detection & Orientation Pipeline"
echo " Reproduction Script"
echo "==========================================="

echo "[1/4] Preparing OBB Datasets for 5-Fold Cross Validation..."
python scripts/prepare_obb_data.py

echo "[2/4] Training YOLOv8 OBB Models (100 Epochs x 5 Folds)..."
python scripts/train_yolov8_obb.py

echo "[3/4] Extracting Deep Features & Training ResNet-18 SVM Classifiers..."
python scripts/train_resnet_svm.py

echo "[4/4] Evaluating the Full Pipeline (YOLO -> MobileSAM -> OpenCV -> ResNet)..."
python scripts/run_sam_eval.py

echo "==========================================="
echo " Reproduction Complete."
echo " To view the results visually, run:"
echo "   streamlit run app.py"
echo "==========================================="
