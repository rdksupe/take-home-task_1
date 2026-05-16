import os
from ultralytics import YOLO

BASE_DIR = 'datasets_pose'
RUNS_DIR = 'models/runs_pose'
EPOCHS = 100
BATCH_SIZE = 16

os.makedirs(RUNS_DIR, exist_ok=True)

for fold in range(5):
    print(f"\n--- Training Fold {fold} ---")
    fold_dir = os.path.join(BASE_DIR, f'fold_{fold}')
    yaml_path = os.path.join(fold_dir, 'data.yaml')
    
    model = YOLO('yolo11s-pose.pt')
    model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=640,
        batch=BATCH_SIZE,
        device=0,
        augment=True,
        degrees=180,
        pose=25.0,
        cos_lr=True,
        project=RUNS_DIR,
        name=f'fold_{fold}',
        verbose=False
    )
print("5-Fold Pose Model Training Complete.")
