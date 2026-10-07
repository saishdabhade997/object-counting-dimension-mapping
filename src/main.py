"""Live object counting + dimension mapping.

Pipeline: camera -> [lens undistort] -> [perspective warp] -> detect (+split) -> mm.

Keys:  c = calibrate on reference object (no-warp mode)   w = re-lock perspective
       s = save snapshot   t = toggle mask view   q = quit
"""
import argparse
import os
import time
import cv2
from src.detector import detect_objects
from src.calibration import Calibrator
from src.visualize import draw
from src.lens import Undistorter
from src.perspective import Warper


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--image", help="process a single image instead of the camera")
    p.add_argument("--ref-mm", type=float, default=85.6, help="reference LONG side in mm (card=85.6)")
    p.add_argument("--ref-pos", choices=["leftmost", "largest"], default="leftmost")
    p.add_argument("--min-area", type=int, default=500)
    p.add_argument("--method", choices=["otsu", "adaptive"], default="otsu")
    p.add_argument("--light-objects", action="store_true")
    p.add_argument("--calib-file", default="calibration.json")
    # enhancements
    p.add_argument("--lens", help="camera_calib.npz from `python -m src.lens calibrate`")
    p.add_argument("--warp", choices=["none", "aruco", "manual"], default="none")
    p.add_argument("--table-mm", type=float, nargs=2, default=[300, 200], metavar=("W", "H"),
                   help="distance between marker centres / table corners in mm")
    p.add_argument("--warp-scale", type=float, default=4.0, help="pixels per mm of the warped view")
    p.add_argument("--split", action="store_true", help="split touching objects (watershed)")
    p.add_argument("--split-ratio", type=float, default=0.7)
    return p.parse_args()


def pick_reference(dets, mode):
    if not dets:
        return None
    return 0 if mode == "leftmost" else max(range(len(dets)), key=lambda i: dets[i].area_px)


def make_warper(frame, a):
    fn = Warper.from_aruco if a.warp == "aruco" else Warper.from_clicks
    return fn(frame, a.table_mm[0], a.table_mm[1], a.warp_scale)


def open_camera(a):
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(a.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    return cap


def main():
    a = parse_args()
    undist = Undistorter.load(a.lens) if a.lens else None
    calib = Calibrator()
    if a.warp == "none":
        try:
            calib = Calibrator.load(a.calib_file)
        except Exception:
            pass
    kw = dict(min_area=a.min_area, method=a.method, invert=not a.light_objects,
              split=a.split, split_ratio=a.split_ratio)

    if a.image:
        frame = cv2.imread(a.image)
        frame = undist.apply(frame) if undist else frame
        ref = None
        if a.warp != "none":
            w = make_warper(frame, a)
            if w is None:
                raise SystemExit("Perspective lock failed (need ArUco IDs 0-3 visible).")
            frame, calib = w.warp(frame), Calibrator(w.px_per_mm)
        dets, _ = detect_objects(frame, **kw)
        if a.warp == "none":
            ref = pick_reference(dets, a.ref_pos)
            if ref is not None:
                calib.calibrate(dets[ref], a.ref_mm)
        for i, d in enumerate(dets):
            l, wd = calib.measure(d) if calib.is_calibrated else (0, 0)
            print(f"Object {i + 1}: {l:.1f} x {wd:.1f} mm")
        print(f"Total count: {len(dets)}")
        cv2.imshow("result", draw(frame, dets, calib, ref))
        cv2.waitKey(0)
        return

    cap = open_camera(a)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera")
    warper, relock, tried_manual = None, a.warp != "none", False
    show_mask, prev = False, time.time()
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        frame = undist.apply(raw) if undist else raw

        if a.warp != "none":
            if relock and (a.warp == "aruco" or not tried_manual or relock == "force"):
                tried_manual = True
                w = make_warper(frame, a)
                if w is not None:
                    warper, calib = w, Calibrator(w.px_per_mm)
                    print(f"Perspective locked ({a.table_mm[0]}x{a.table_mm[1]} mm @ {w.px_per_mm} px/mm)")
                relock = relock if (w is None and a.warp == "aruco") else False
            if warper is None:
                msg = "Searching for ArUco markers 0-3..." if a.warp == "aruco" else "Press 'w' to click corners"
                cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Object Counting & Dimension Mapping", frame)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    break
                if k == ord("w"):
                    relock = "force"
                continue
            frame = warper.warp(frame)

        dets, mask = detect_objects(frame, **kw)
        ref = pick_reference(dets, a.ref_pos) if a.warp == "none" else None
        now = time.time()
        fps, prev = 1 / max(now - prev, 1e-6), now
        view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if show_mask else draw(frame, dets, calib, ref, fps)
        cv2.imshow("Object Counting & Dimension Mapping", view)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("c") and ref is not None:
            calib.calibrate(dets[ref], a.ref_mm)
            calib.save(a.calib_file)
            print(f"Calibrated: {calib.pixels_per_mm:.3f} px/mm")
        elif k == ord("w") and a.warp != "none":
            relock = "force"
        elif k == ord("s"):
            name = f"snapshot_{int(time.time())}.png"
            cv2.imwrite(name, draw(frame, dets, calib, ref, fps))
            print("Saved", name)
        elif k == ord("t"):
            show_mask = not show_mask
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
