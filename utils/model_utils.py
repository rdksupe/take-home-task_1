import torch
import torchvision.transforms as T
import torchvision.models as models
import cv2
from utils.image_utils import extract_patch

# Lazy-loaded globals (initialized on first use, not on import)
_resnet = None
_transform = None
_device = None

def get_resnet_model():
    """Load pre-trained ResNet-18 (truncated before FC) for feature extraction."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
    resnet = resnet.to(device)
    resnet.eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return resnet, transform, device

def _ensure_loaded():
    """Lazy-initialize the ResNet model on first use."""
    global _resnet, _transform, _device
    if _resnet is None:
        _resnet, _transform, _device = get_resnet_model()

@torch.no_grad()
def extract_features(patch):
    """Extract a 512-dim feature vector from a patch using ResNet-18."""
    _ensure_loaded()
    if len(patch.shape) == 2:
        patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
    t_patch = _transform(patch).unsqueeze(0).to(_device)
    feat = _resnet(t_patch).flatten().cpu().numpy()
    return feat

def resolve_180_resnet(cx, cy, pred_angle, img, svm_clf):
    """Resolve 180-degree ambiguity by classifying patches at both ends of the PCA axis.

    Extracts patches at pred_angle and pred_angle+180, classifies which is the
    'Tab' (class 1), and returns the angle pointing toward the tab.
    """
    results = []
    for offset in [0, 180]:
        ang = (pred_angle + offset) % 360
        patch = extract_patch(cx, cy, ang, img)
        if patch is not None:
            feat = extract_features(patch)
            prob = svm_clf.predict_proba([feat])[0][1]
            results.append((prob, ang))
    if not results:
        return pred_angle
    results.sort(key=lambda x: x[0], reverse=True)
    return results[0][1]
