"""Measure accuracy against ground truth (this produces the real '±X mm' for your resume).

Usage:
  python -m src.evaluate --image samples/test.jpg --ref-mm 85.6 --truth samples/truth.csv

truth.csv (objects ordered LEFT -> RIGHT, include the reference object as row 1):
  length_mm,width_mm
  85.6,54.0
  60.0,40.0
"""
import argparse
import csv
import cv2
import numpy as np
from src.detector import detect_objects
from src.calibration import Calibrator


def evaluate(image_path, ref_mm, truth_csv, **kw):
    frame = cv2.imread(image_path)
    dets, _ = detect_objects(frame, **kw)
    with open(truth_csv) as f:
        truth = [(float(r["length_mm"]), float(r["width_mm"])) for r in csv.DictReader(f)]
    if len(dets) != len(truth):
        print(f"WARNING: detected {len(dets)} objects but truth has {len(truth)}")
    cal = Calibrator()
    cal.calibrate(dets[0], ref_mm)
    errs = []
    for i, (d, (tl, tw)) in enumerate(zip(dets, truth)):
        l, w = cal.measure(d)
        errs += [abs(l - tl), abs(w - tw)]
        print(f"#{i + 1}: measured {l:.2f}x{w:.2f} | truth {tl}x{tw} | err {abs(l-tl):.2f}, {abs(w-tw):.2f} mm")
    errs = np.array(errs[2:])  # exclude reference object (error is zero by construction)
    if errs.size:
        print(f"\nMAE = {errs.mean():.2f} mm | max error = {errs.max():.2f} mm | "
              f"std = {errs.std():.2f} mm  ->  report as ±{errs.max():.1f} mm")
    return errs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--ref-mm", type=float, required=True)
    p.add_argument("--truth", required=True)
    a = p.parse_args()
    evaluate(a.image, a.ref_mm, a.truth)
