"""Lens distortion correction (checkerboard calibration).

  python -m src.lens capture   --camera 1 --out calib_images        # SPACE saves a view
  python -m src.lens calibrate --images calib_images --cols 9 --rows 6 --square-mm 25
Use the SAME resolution for capture, calibration and the live run.
"""
import argparse
import glob
import os
import cv2
import numpy as np


def _open(index, width, height):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def capture(camera, out_dir, cols, rows, width, height):
    os.makedirs(out_dir, exist_ok=True)
    cap, n = _open(camera, width, height), len(glob.glob(os.path.join(out_dir, "*.png")))
    print("SPACE = save (only when corners are detected), q = quit. Aim for 15-25 varied views:"
          " tilt the board, move it to corners/edges of the frame.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_FAST_CHECK)
        view = frame.copy()
        if found:
            cv2.drawChessboardCorners(view, (cols, rows), corners, found)
        cv2.putText(view, f"saved: {n}  corners: {'OK' if found else 'no'}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("lens capture", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == 32 and found:
            cv2.imwrite(os.path.join(out_dir, f"view_{n:02d}.png"), frame)
            n += 1
    cap.release()
    cv2.destroyAllWindows()


def calibrate(images_dir, cols, rows, square_mm, out="camera_calib.npz"):
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    obj_pts, img_pts, size, used = [], [], None, []
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for path in sorted(glob.glob(os.path.join(images_dir, "*.png")) +
                       glob.glob(os.path.join(images_dir, "*.jpg"))):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows))
        if not found:
            print("skip (no corners):", path)
            continue
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
        obj_pts.append(objp)
        img_pts.append(corners)
        used.append(path)
        size = gray.shape[::-1]
    if len(obj_pts) < 8:
        raise SystemExit(f"Need >= 8 good views, got {len(obj_pts)}")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, size, None, None)

    # per-view reprojection error
    per_view = []
    for o, i, r, t in zip(obj_pts, img_pts, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, r, t, K, dist)
        per_view.append(float(np.sqrt(np.mean((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2))))
    np.savez(out, K=K, dist=dist, size=np.array(size))
    print(f"views used: {len(used)} | RMS reprojection error: {rms:.3f} px "
          f"(good < 0.5) | worst view: {max(per_view):.3f} px")
    print("K =\n", K, "\ndist =", dist.ravel())
    return rms, K, dist, size


class Undistorter:
    def __init__(self, K, dist, size, alpha=0.0):
        self.size = tuple(int(v) for v in size)
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, self.size, alpha, self.size)
        self.map1, self.map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, self.size, cv2.CV_16SC2)

    @classmethod
    def load(cls, path="camera_calib.npz"):
        d = np.load(path)
        return cls(d["K"], d["dist"], d["size"])

    def apply(self, frame):
        if (frame.shape[1], frame.shape[0]) != self.size:
            raise ValueError(f"Frame is {frame.shape[1]}x{frame.shape[0]} but lens calibration was "
                             f"{self.size[0]}x{self.size[1]} - use the same --width/--height.")
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--camera", type=int, default=0)
    c.add_argument("--out", default="calib_images")
    c.add_argument("--width", type=int, default=1280)
    c.add_argument("--height", type=int, default=720)
    k = sub.add_parser("calibrate")
    k.add_argument("--images", default="calib_images")
    k.add_argument("--square-mm", type=float, required=True)
    k.add_argument("--output", default="camera_calib.npz")
    for s in (c, k):
        s.add_argument("--cols", type=int, default=9, help="INNER corners per row")
        s.add_argument("--rows", type=int, default=6, help="INNER corners per column")
    a = p.parse_args()
    if a.cmd == "capture":
        capture(a.camera, a.out, a.cols, a.rows, a.width, a.height)
    else:
        calibrate(a.images, a.cols, a.rows, a.square_mm, a.output)
