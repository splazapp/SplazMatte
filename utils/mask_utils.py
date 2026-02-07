"""Mask visualization utilities for SAM2 annotation overlay."""

import cv2
import numpy as np


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    color: tuple[int, int, int] = (0, 120, 255),
) -> np.ndarray:
    """Overlay a semi-transparent colored mask on an image.

    Args:
        image: RGB uint8 array (H, W, 3).
        mask: Binary mask (H, W), values in {0, 1} or {0, 255}.
        alpha: Blend factor for the mask region.
        color: RGB color tuple for the mask overlay.

    Returns:
        RGB uint8 array (H, W, 3) with mask overlaid.
    """
    out = image.copy()
    binary = mask > 0.5
    color_arr = np.array(color, dtype=np.float32)
    out[binary] = (
        out[binary].astype(np.float32) * (1 - alpha) + color_arr * alpha
    ).astype(np.uint8)

    # Draw contour
    contour_mask = binary.astype(np.uint8)
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (255, 255, 255), 2)
    return out


def draw_points(
    image: np.ndarray,
    points: list[list[int]],
    labels: list[int],
    radius: int = 8,
) -> np.ndarray:
    """Draw positive (green) and negative (red) points on an image.

    Args:
        image: RGB uint8 array (H, W, 3).
        points: List of [x, y] coordinates.
        labels: List of labels (1=positive, 0=negative).
        radius: Circle radius in pixels.

    Returns:
        RGB uint8 array (H, W, 3) with points drawn.
    """
    out = image.copy()
    for (x, y), label in zip(points, labels):
        color = (0, 255, 0) if label == 1 else (255, 0, 0)
        cv2.circle(out, (int(x), int(y)), radius, color, -1)
        cv2.circle(out, (int(x), int(y)), radius, (255, 255, 255), 2)
    return out


def draw_frame_number(image: np.ndarray, frame_idx: int) -> np.ndarray:
    """Draw frame index number on the top-right corner of an image.

    Args:
        image: RGB uint8 array (H, W, 3).
        frame_idx: Frame index to display.

    Returns:
        RGB uint8 array (H, W, 3) with frame number drawn.
    """
    out = image.copy()
    text = f"#{frame_idx}"
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Scale font relative to image height
    h, w = out.shape[:2]
    font_scale = max(0.5, h / 800)
    thickness = max(1, int(font_scale * 2))

    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    margin = int(10 * font_scale)

    x = w - tw - margin
    y = th + margin

    # Draw background rectangle for readability
    cv2.rectangle(
        out,
        (x - margin // 2, y - th - margin // 2),
        (x + tw + margin // 2, y + baseline + margin // 2),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(out, text, (x, y), font, font_scale, (255, 255, 255), thickness)
    return out
