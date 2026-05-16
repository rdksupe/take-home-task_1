import math

# Patch extraction parameters:
#   R = 15 pixels: offset from center along the axis to sample patch centers.
#     Chosen empirically — large enough to capture distinct tab/joint texture,
#     small enough to stay within the tube body.
#   S = 20 pixels: half-size of the extracted patch (40x40 total).
#     Provides sufficient context for ResNet-18 feature extraction.

PATCH_OFFSET_R = 15
PATCH_HALF_SIZE = 20

def extract_patch(cx, cy, angle, img, S=PATCH_HALF_SIZE, R=PATCH_OFFSET_R):
    """Extract a (2S x 2S) patch centered R pixels from (cx, cy) along the given angle."""
    rad = math.radians(angle)
    px = int(cx + R * math.cos(rad))
    py = int(cy - R * math.sin(rad))
    if 0 <= py-S and py+S < img.shape[0] and 0 <= px-S and px+S < img.shape[1]:
        return img[py-S:py+S, px-S:px+S]
    return None

def get_patches_from_gt(cx, cy, angle_deg, img, S=PATCH_HALF_SIZE):
    """Extract tab and joint patches using ground-truth angle (for SVM training)."""
    R = PATCH_OFFSET_R
    rad_tab = math.radians(angle_deg)
    rad_joint = math.radians((angle_deg + 180) % 360)

    tab_x = int(cx + R * math.cos(rad_tab))
    tab_y = int(cy - R * math.sin(rad_tab))
    joint_x = int(cx + R * math.cos(rad_joint))
    joint_y = int(cy - R * math.sin(rad_joint))

    tab_patch, joint_patch = None, None
    if 0 <= tab_y-S and tab_y+S < img.shape[0] and 0 <= tab_x-S and tab_x+S < img.shape[1]:
        tab_patch = img[tab_y-S:tab_y+S, tab_x-S:tab_x+S]
    if 0 <= joint_y-S and joint_y+S < img.shape[0] and 0 <= joint_x-S and joint_x+S < img.shape[1]:
        joint_patch = img[joint_y-S:joint_y+S, joint_x-S:joint_x+S]

    return tab_patch, joint_patch
