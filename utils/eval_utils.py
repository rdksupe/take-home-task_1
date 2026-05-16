import cv2
import math
import numpy as np
from utils.model_utils import resolve_180_resnet

# Bounding box expansion (pixels): provides SAM with slightly larger context
# around the YOLO detection to ensure the full tab geometry is captured.
BBOX_EXPANSION_PX = 5

# True-positive matching threshold (pixels): maximum distance between a
# predicted center and a ground-truth center to count as a correct match.
# Set to ~half the typical tube diameter.
TP_MATCH_THRESHOLD_PX = 25.0

def get_pca_angle(mask):
    """Derive the major-axis orientation of a binary mask using image moments (PCA).

    Returns (cx, cy, angle) where angle is in [0, 180) degrees, or (None, None, None)
    if the mask is degenerate.
    """
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None
    c = max(contours, key=cv2.contourArea)
    M = cv2.moments(c)
    if M['m00'] != 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
        mu20 = M['mu20'] / M['m00']
        mu02 = M['mu02'] / M['m00']
        mu11 = M['mu11'] / M['m00']
        theta = 0.5 * math.atan2(2 * mu11, mu20 - mu02)
        pca_ang = (-math.degrees(theta) + 360) % 180
        return cx, cy, pca_ang
    return None, None, None

def evaluate_single_image(img, obb_model, sam_model, svm_clf):
    """Run the full pipeline on a single image: YOLO -> SAM -> PCA -> ResNet SVM.

    Returns:
        pred_centers: list of [cx, cy] for each detected tube.
        pred_angles:  list of final orientation angles in [0, 360) degrees.
    """
    results = obb_model(img, verbose=False)

    boxes_to_process = []

    if hasattr(results[0], 'obb') and results[0].obb is not None and len(results[0].obb) > 0:
        for box in results[0].obb:
            xyxyxyxy = box.xyxyxyxy.cpu().numpy()[0]
            x_min, x_max = np.min(xyxyxyxy[:, 0]), np.max(xyxyxyxy[:, 0])
            y_min, y_max = np.min(xyxyxyxy[:, 1]), np.max(xyxyxyxy[:, 1])
            boxes_to_process.append([x_min, y_min, x_max, y_max])
    elif hasattr(results[0], 'boxes') and results[0].boxes is not None and len(results[0].boxes) > 0:
        for box in results[0].boxes.xyxy.cpu().numpy():
            x_min, y_min, x_max, y_max = box
            boxes_to_process.append([x_min, y_min, x_max, y_max])

    if not boxes_to_process:
        return [], []

    pred_centers = []
    pred_angles = []

    for box in boxes_to_process:
        x_min, y_min, x_max, y_max = box
        bbox = [
            int(x_min - BBOX_EXPANSION_PX),
            int(y_min - BBOX_EXPANSION_PX),
            int(x_max + BBOX_EXPANSION_PX),
            int(y_max + BBOX_EXPANSION_PX),
        ]

        sam_results = sam_model.predict(img, bboxes=[bbox], verbose=False)

        if len(sam_results[0].masks) > 0:
            mask = sam_results[0].masks.data[0].cpu().numpy()
            if mask.shape != img.shape[:2]:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

            cx, cy, pca_ang = get_pca_angle(mask)
            if pca_ang is not None:
                final_ang = resolve_180_resnet(cx, cy, pca_ang, img, svm_clf)

                pred_centers.append([cx, cy])
                pred_angles.append(final_ang)

    return pred_centers, pred_angles
