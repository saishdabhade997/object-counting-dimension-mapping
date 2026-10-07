import cv2
import numpy as np
from src.detector import detect_objects
from src.calibration import Calibrator
from src.lens import calibrate, Undistorter
from src.perspective import Warper, make_marker


# ---------- watershed ----------
def two_touching_circles():
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.circle(img, (140, 150), 50, (20, 20, 20), -1)
    cv2.circle(img, (225, 150), 50, (20, 20, 20), -1)   # overlap -> one blob
    return img


def test_touching_objects_merge_without_split():
    dets, _ = detect_objects(two_touching_circles(), split=False)
    assert len(dets) == 1


def test_watershed_splits_touching_objects():
    dets, _ = detect_objects(two_touching_circles(), split=True)
    assert len(dets) == 2
    assert all(abs(d.long_px - 100) < 12 for d in dets)  # each ~100 px diameter


def test_split_does_not_break_single_rectangles():
    img = np.full((300, 400, 3), 255, np.uint8)
    cv2.rectangle(img, (50, 50), (200, 120), (20, 20, 20), -1)
    cv2.rectangle(img, (250, 150), (350, 250), (20, 20, 20), -1)
    dets, _ = detect_objects(img, split=True)
    assert len(dets) == 2


# ---------- perspective ----------
def camera_view(scale=2.0):
    """Top-down scene (table 300x200 mm, markers at corners) seen from a tilted camera."""
    m = 100
    canvas = np.full((int(200 * scale) + 2 * m, int(300 * scale) + 2 * m, 3), 255, np.uint8)
    centers = [(m, m), (m + 600, m), (m + 600, m + 400), (m, m + 400)]
    for i, (cx, cy) in enumerate(centers):
        mk = cv2.cvtColor(make_marker(i, 90), cv2.COLOR_GRAY2BGR)
        canvas[cy - 45:cy + 45, cx - 45:cx + 45] = mk
    # objects with known size: 80x50 mm and 40x40 mm
    cv2.rectangle(canvas, (m + 100, m + 100), (m + 100 + 160, m + 100 + 100), (30, 30, 30), -1)
    cv2.rectangle(canvas, (m + 350, m + 200), (m + 350 + 80, m + 200 + 80), (30, 30, 30), -1)
    h, w = canvas.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[60, 40], [w - 20, 90], [w - 90, h - 10], [20, h - 70]])  # perspective tilt
    return cv2.warpPerspective(canvas, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderValue=(255, 255, 255))


def test_aruco_homography_recovers_metric_size():
    view = camera_view()
    warper = Warper.from_aruco(view, 300, 200, px_per_mm=4.0)
    assert warper is not None
    top = warper.warp(view)
    dets, _ = detect_objects(top, min_area=2000)
    assert len(dets) == 2
    cal = Calibrator(warper.px_per_mm)
    l, w = cal.measure(dets[0])
    assert abs(l - 80) < 1.5 and abs(w - 50) < 1.5
    l, w = cal.measure(dets[1])
    assert abs(l - 40) < 1.5 and abs(w - 40) < 1.5


# ---------- lens calibration ----------
def test_lens_calibration_recovers_focal_length(tmp_path):
    cols, rows, sq = 9, 6, 30
    board = np.full(((rows + 1) * sq + 2 * sq, (cols + 1) * sq + 2 * sq), 255, np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y, x = sq + r * sq, sq + c * sq
                board[y:y + sq, x:x + sq] = 0
    bh, bw = board.shape
    K = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], np.float64)
    corners3d = np.float32([[0, 0, 0], [bw, 0, 0], [bw, bh, 0], [0, bh, 0]]) - [bw / 2, bh / 2, 0]
    rng = np.random.default_rng(1)
    for i in range(14):
        rvec = rng.uniform(-0.5, 0.5, 3)
        tvec = np.array([rng.uniform(-40, 40), rng.uniform(-30, 30), rng.uniform(650, 850)])
        pts, _ = cv2.projectPoints(corners3d, rvec, tvec, K, None)
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [bw, 0], [bw, bh], [0, bh]]),
                                         pts.reshape(4, 2).astype(np.float32))
        img = cv2.warpPerspective(board, Hm, (640, 480), borderValue=255)
        cv2.imwrite(str(tmp_path / f"v{i:02d}.png"), img)
    rms, Kest, dist, size = calibrate(str(tmp_path), cols, rows, sq, out=str(tmp_path / "c.npz"))
    assert rms < 1.0
    assert abs(Kest[0, 0] - 800) / 800 < 0.05
    und = Undistorter.load(str(tmp_path / "c.npz"))
    assert und.apply(np.zeros((480, 640, 3), np.uint8)).shape == (480, 640, 3)
