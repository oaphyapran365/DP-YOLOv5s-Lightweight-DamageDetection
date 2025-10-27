
"""
utils_colors.py
Centralized color policy for visualization overlays.

We assign fixed colors (in OpenCV BGR format) to each damage category.
These colors are chosen for field usability and quick triage:

    - severe damage   -> red
         "urgent / high priority / likely collapse"

    - moderate damage -> orange
         "significant structural impact, but not total failure"

    - light damage    -> blue
         "cosmetic / partial roof issues / cracked facade"

    - no damage       -> sky-ish / pale
         "appears intact"

Keeping this mapping consistent across image and video inference
ensures responders see the same risk language in every demo,
frame capture, or dashboard screenshot.

Note: OpenCV expects colors as (B, G, R).


# ---------------------------------------------------------------------------
# 💡 HOW TO USE THIS UTILITY
# ---------------------------------------------------------------------------
# In your detection or visualization script (e.g., detect_image.py / detect_video.py),
# import the function:
#
#     from utils_colors import get_color
#
# Then, inside your detection loop where you draw bounding boxes:
#
#     # Get the fixed color based on predicted label
#     color = get_color(label)
#
#     # Draw bounding box using YOLOv5's Annotator
#     annotator.box_label((x1, y1, x2, y2), label, color=color)
#
# This ensures that all your results — images, videos, screenshots — use the
# same consistent color scheme across the entire project.
# ---------------------------------------------------------------------------
"""


# Fixed BGR colors for each canonical label we predict.
FIXED_COLORS_BGR = {
    "severe damage":   (0,   0, 255),   # red
    "moderate damage": (0, 165, 255),   # orange
    "light damage":    (255, 0,   0),   # blue
    "no damage":       (235, 206, 135), # pale/sky-like highlight
}


def get_color(label: str):
    """
    Return the BGR tuple for a given class label (case-insensitive).
    Falls back to a default if label is unknown.

    Args:
        label (str): e.g. "severe damage", "light damage", ...

    Returns:
        (B, G, R) tuple of ints in [0,255]
    """
    if not isinstance(label, str):
        return (255, 255, 255)  # white fallback

    key = label.strip().lower()
    return FIXED_COLORS_BGR.get(key, (255, 255, 255))  # default = white box

