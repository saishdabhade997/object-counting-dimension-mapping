"""Perspective (homography) correction -> true top-down, metric view.

Place 4 ArUco markers (DICT_4X4_50, IDs 0,1,2,3 = TL,TR,BR,BL) at the corners of the work
area and measure the distance between marker CENTERS (--table-mm WIDTH HEIGHT).
After warping, 1 pixel = 1/px_per_mm mm everywhere, so no reference object is needed.
Objects must lie inside the rectangle spanned by the marker centers.

  python -m src.perspective make-markers      # writes marker_0..3.png to print
"""
import sys
import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)


def make_marker(marker_id, side=400):
    return cv2.aruco.generateImageMarker(DICT, marker_id, side)


def _diag_intersection(c):
    """Exact projected centre of a square from its 4 image corners (diagonal intersection)."""
    p0, p1, p2, p3 = [np.asarray(p, np.float64) for p in c]
    A = np.array([p2 - p0, p1 - p3]).T
    t = np.linalg.solve(A, p1 - p0)
    return p0 + t[0] * (p2 - p0)


def detect_marker_centers(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    detector = cv2.aruco.ArucoDetector(DICT, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): _diag_intersection(c.reshape(4, 2)) for i, c in zip(ids.ravel(), corners)}


class Warper:
    def __init__(self, src_pts, table_w_mm, table_h_mm, px_per_mm=4.0):
        self.px_per_mm = float(px_per_mm)
        w, h = table_w_mm * px_per_mm, table_h_mm * px_per_mm
        dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        self.H = cv2.getPerspectiveTransform(np.float32(src_pts), dst)
        self.size = (int(np.ceil(w)), int(np.ceil(h)))

    def warp(self, frame):
        return cv2.warpPerspective(frame, self.H, self.size, flags=cv2.INTER_LINEAR)

    @classmethod
    def from_aruco(cls, frame, table_w_mm, table_h_mm, px_per_mm=4.0):
        centers = detect_marker_centers(frame)
        if not all(i in centers for i in range(4)):
            return None
        return cls([centers[i] for i in range(4)], table_w_mm, table_h_mm, px_per_mm)

    @classmethod
    def from_clicks(cls, frame, table_w_mm, table_h_mm, px_per_mm=4.0):
        """Fallback without markers: click the 4 table corners TL, TR, BR, BL."""
        pts, win = [], "click TL, TR, BR, BL (ESC cancels)"

        def on_mouse(event, x, y, *_):
            if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
                pts.append((x, y))

        cv2.namedWindow(win)
        cv2.setMouseCallback(win, on_mouse)
        while len(pts) < 4:
            view = frame.copy()
            for p in pts:
                cv2.circle(view, p, 6, (0, 0, 255), -1)
            cv2.imshow(win, view)
            if cv2.waitKey(20) & 0xFF == 27:
                cv2.destroyWindow(win)
                return None
        cv2.destroyWindow(win)
        return cls(pts, table_w_mm, table_h_mm, px_per_mm)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "make-markers":
    for i in range(4):
        img = cv2.copyMakeBorder(make_marker(i), 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite(f"marker_{i}.png", img)
    print("Wrote marker_0..3.png - print at the same size, measure centre-to-centre distances.")
