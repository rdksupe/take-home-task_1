import math

def angle_diff(a1, a2):
    """Calculate the minimum angular difference between two angles in degrees."""
    diff = abs(a1 - a2) % 360
    return min(diff, 360 - diff)
