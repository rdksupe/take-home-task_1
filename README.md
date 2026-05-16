# Tube Detection & Orientation Pipeline

## 1.  Overview

I approached the take-home using a multi-stage pipeline, achieving a **4.59° Mean Angle Error** with **100% Precision and 100% Recall** across all 371 ground-truth tubes. 

## 2. Pipeline Architecture
1. **Detection (YOLOv8 OBB):** Detects tubes and outputs Oriented Bounding Boxes, eliminating background noise. YOLOv8 is chosen over YOLOv26 for 100% recall (see report for trade-off analysis).
2. **Sub-Pixel Masking (MobileSAM):** Uses the YOLO box as a prompt to segment a pixel-perfect binary mask of the transparent plastic.
3. **Axis Calculation (OpenCV Moments):** Calculates the PCA major axis of the SAM mask, giving a mathematically precise angle in [0, 180°).
4. **Symmetry Resolution (ResNet-18 + SVM):** Extracts patches from both ends of the axis and classifies them to resolve the 180° front/back ambiguity.

> **📄 For detailed methodology, mathematical derivations, failure analysis, and comprehensive evaluation, please refer to the included [`report.pdf`](./report.pdf).**

## 3. Directory Structure
```
.
├── app.py                   # Streamlit visualization application
├── reproduce.sh             # End-to-end reproduction script
├── annotations.csv          # Ground truth data
├── images/                  # Raw RGB images (640x480)
├── models/                  # Pre-trained model weights (see download_models.sh)
├── documentation/           # Scaling laws, math explanations, and analysis
├── scripts/                 # Training, data prep, and evaluation scripts
└── utils/                   # Shared utility modules
```

## 4. How to Reproduce
To reproduce the full pipeline from scratch (dataset generation, YOLO training, SVM training, evaluation):

```bash
pip install -r requirements.txt
chmod +x reproduce.sh
./reproduce.sh
```
*Note: The YOLOv8 step trains 5 models for 100 epochs each. A CUDA GPU is highly recommended.*

To run with pre-trained weights only:
```bash
python scripts/run_sam_eval.py
```

## 5. Results (5-Fold Cross-Validation)

### Per-Fold Breakdown

| Fold | TPs | FPs | GT  | Recall | MAE    | Median | < 10°   |
|------|-----|-----|-----|--------|--------|--------|---------|
| 0    | 76  | 0   | 76  | 1.0000 | 4.78°  | 2.95°  | 89.47%  |
| 1    | 73  | 0   | 73  | 1.0000 | 3.72°  | 2.15°  | 91.78%  |
| 2    | 78  | 0   | 78  | 1.0000 | 4.33°  | 2.62°  | 88.46%  |
| 3    | 67  | 0   | 67  | 1.0000 | 5.36°  | 3.00°  | 86.57%  |
| 4    | 77  | 0   | 77  | 1.0000 | 4.84°  | 2.55°  | 80.52%  |

### Aggregate Results (371 GT Tubes)

| Metric              | Value              |
|---------------------|--------------------|
| Matched TPs         | 371                |
| False Positives     | 0                  |
| Undetected          | 0                  |
| **Precision**       | **1.0000**         |
| **Recall**          | **1.0000 (371/371)** |
| **F1 Score**        | **1.0000**         |

### Orientation Accuracy

| Metric                | Value     |
|-----------------------|-----------|
| **Mean Angle Error**  | **4.59°** |
| **Median Angle Error**| **2.75°** |
| Errors < 5°           | 73.05%    |
| Errors < 10°          | 87.33%    |
| Errors < 22°          | 96.77%    |
| Flip Failures (≥ 90°) | 0.00%    |

<details>
<summary>Error Distribution Breakdown</summary>

| Bucket    | Percentage |
|-----------|------------|
| 0° – 1°  | 19.68%     |
| 1° – 2°  | 20.49%     |
| 2° – 3°  | 13.48%     |
| 3° – 4°  | 10.78%     |
| 4° – 5°  | 8.63%      |

</details>

## 6. Visualizing the Results
A Streamlit dashboard lets you visually inspect the pipeline predictions against ground truth:

```bash
streamlit run app.py
```


## 7. Pipeline in Action

The collage below shows 8 random images progressing simultaneously through the full pipeline — from raw detection to final oriented output:

![Pipeline Collage](./outputs/collage_montage.gif)

## 8. Use of AI Tools
AI coding assistants were used throughout development for:
- **Code scaffolding:** Boilerplate for YOLO training loops, data preparation, and Streamlit UI.
- **Documentation:** Drafting LaTeX report structure and markdown documentation.

All technical decisions (pipeline architecture, the SAM + PCA moments approach, the ResNet-SVM flip resolution strategy) and the written analysis are my own.
