"""Train YOLOv8 OBB models for 5-fold cross-validation.

Uses YOLOv8n-OBB (nano) for 100 epochs per fold.
YOLOv8 is chosen over YOLOv26 because it achieves 100% recall (371/371 tubes),
whereas YOLOv26 drops 4 tubes under extreme glare/shadow.

Weights are saved to models/runs_yolov8/fold_{N}/weights/best.pt.
"""
from ultralytics import YOLO
import os

os.makedirs('models/runs_yolov8', exist_ok=True)

for fold in range(5):
    print(f"--- Training YOLOv8 OBB Fold {fold} ---")
    model = YOLO('models/yolov8n-obb.pt', task='obb')

    results = model.train(
        data=f'datasets_obb/fold_{fold}/dataset.yaml',
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,
        workers=1,
        verbose=False,
        plots=False,
        save=True,
        project='models/runs_yolov8',
        name=f'fold_{fold}'
    )

print("YOLOv8 OBB training complete for all 5 folds.")
