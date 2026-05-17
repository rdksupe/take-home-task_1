#!/bin/bash
# ===========================================
#  Tube Detection & Orientation Pipeline
#  Evaluation with Provided Weights
# ===========================================
set -e

echo "==========================================="
echo " [1/5] Preparing Datasets (5-Fold CV)..."
python scripts/prepare_obb_data.py
python scripts/prepare_pose_data.py

echo "   (Skipping full retrain. Using provided weights for evaluation.)"

echo " [3/5] Benchmarking Baseline Logic (SAM-PCA + SVM)..."
python scripts/run_sam_eval.py

echo " [4/5] Benchmarking Final Refined Logic (SAM-PCA + Pose)..."
python scripts/run_final_eval.py

echo " [5/5] Launching Visual Audit Dashboard..."
echo " To view results, run: streamlit run app.py"
echo "==========================================="
