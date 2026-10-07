import cv2
import numpy as np
from src.detector import detect_objects
from src.calibration import Calibrator

SCALE = 5.0  # px per mm


def synthetic(rects_mm):
    img = np.full((600, 900, 3), 255, np.uint8)
    x = 40
    for (l, w) in rects_mm:
        cv2.rectangle(img, (x, 100), (x + int(l * SCALE), 100 + int(w * SCALE)), (30, 30, 30), -1)
        x += int(l * SCALE) + 40
    noise = np.random.default_rng(0).integers(0, 15, img.shape, dtype=np.uint8)
    return cv2.subtract(img, noise)


def test_counts_objects():
    dets, _ = detect_objects(synthetic([(60, 40), (50, 30), (40, 20)]))
    assert len(dets) == 3


def test_dimension_accuracy():
    dets, _ = detect_objects(synthetic([(60, 40), (50, 30), (40, 20)]))
    cal = Calibrator()
    cal.calibrate(dets[0], 60)
    l, w = cal.measure(dets[1])
    assert abs(l - 50) < 1.0 and abs(w - 30) < 1.0


def test_filters_small_noise():
    img = synthetic([(60, 40)])
    cv2.circle(img, (700, 400), 3, (0, 0, 0), -1)
    dets, _ = detect_objects(img, min_area=500)
    assert len(dets) == 1
