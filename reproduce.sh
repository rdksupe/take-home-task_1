#!/bin/bash
# ===========================================
#  Tube Detection & Orientation Pipeline
#  Full Reproduction Script ( Refined)
# ===========================================
set -e

echo "==========================================="
echo " [1/5] Preparing Datasets (5-Fold CV)..."
python scripts/prepare_pose_data.py

echo " [2/5] Training YOLO11s-Pose Models (100 Epochs x 5 Folds)..."
python scripts/train_pose_model.py

echo " [3/5] Benchmarking Original Logic (SAM-PCA + SVM)..."
python scripts/run_sam_eval.py

echo " [4/5] Benchmarking Final Refined Logic (SAM-PCA + Pose)..."
python scripts/run_final_eval.py

echo " [5/5] Launching Visual Audit Dashboard..."
echo " To view results, run: streamlit run app.py"
echo "==========================================="
