"""Object detection & counting: blur -> threshold -> morphology -> contours -> filtering."""
from dataclasses import dataclass
import cv2
import numpy as np
from src.splitter import split_touching


@dataclass
class Detection:
    contour: np.ndarray
    box: np.ndarray          # 4x2 corner points of the min-area rectangle
    center: tuple            # (cx, cy) in pixels
    long_px: float           # longer side of min-area rect (pixels)
    short_px: float          # shorter side of min-area rect (pixels)
    area_px: float


def preprocess(frame, blur_ksize=5):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    k = blur_ksize | 1  # kernel must be odd
    return cv2.GaussianBlur(gray, (k, k), 0)


def segment(gray, method="otsu", invert=True, block_size=35, c=5, morph_ksize=5):
    """Return a clean binary mask. invert=True -> dark objects on light background."""
    flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    if method == "otsu":
        _, mask = cv2.threshold(gray, 0, 255, flag + cv2.THRESH_OTSU)
    elif method == "adaptive":
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     flag, block_size | 1, c)
    else:
        raise ValueError(f"Unknown threshold method: {method}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ksize, morph_ksize))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)    # remove speckle noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)   # fill small holes
    return mask


def detect_objects(frame, min_area=500, max_area_ratio=0.5, method="otsu",
                   invert=True, ignore_border=True, blur_ksize=5,
                   split=False, split_ratio=0.7):
    """Detect objects; returns (detections sorted left->right, binary mask)."""
    h, w = frame.shape[:2]
    mask = segment(preprocess(frame, blur_ksize), method=method, invert=invert)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if split:  # separate touching objects (noise is dropped first to save time)
        expanded = []
        for c in contours:
            if cv2.contourArea(c) >= min_area:
                expanded += split_touching(c, split_ratio)
        contours = expanded

    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area_ratio * w * h:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if ignore_border and (x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1):
            continue  # partially visible objects give wrong dimensions
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), _ = rect
        detections.append(Detection(
            contour=cnt, box=cv2.boxPoints(rect).astype(int), center=(cx, cy),
            long_px=max(rw, rh), short_px=min(rw, rh), area_px=area))

    detections.sort(key=lambda d: d.center[0])
    return detections, mask
