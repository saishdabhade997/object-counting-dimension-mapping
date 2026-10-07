# Object Counting & Dimension Mapping
Real-time computer vision pipeline (Python + OpenCV) that counts objects from a live camera feed and measures their real-world width/height using a reference-object scale calibration.

## Pipeline
Frame -> grayscale + Gaussian blur -> Otsu/adaptive threshold -> morphological open/close (noise filtering) -> contours -> area/border filtering -> min-area rectangle -> pixels -> mm via `pixels_per_mm`.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest            # unit tests on synthetic data
python -m src.main          # live camera
python -m src.main --image samples/test.jpg --ref-mm 85.6
```
Keys: `c` calibrate, `s` snapshot, `t` mask view, `q` quit.

## Physical setup (matters for accuracy)
- Plain, contrasting background (white paper + dark objects, or use `--light-objects`).
- Camera mounted perpendicular to the surface; even lighting, no shadows.
- Reference object (e.g. credit card, 85.6 mm) placed as the **leftmost** object, on the same plane as the objects to measure.

## Accuracy
Run `python -m src.evaluate --image ... --ref-mm 85.6 --truth samples/truth.csv` against objects measured with a ruler/caliper. Results:

| Test | Objects | MAE (mm) | Max error (mm) |
|------|---------|----------|----------------|
| TODO | | | |

## Limitations / future work
Lens distortion (camera calibration), touching objects (watershed), perspective correction (homography), tall objects (different plane than reference).

## Advanced modules
| Module | Purpose | How to run |
|---|---|---|
| `src/lens.py` | Lens distortion correction (checkerboard) | `python -m src.lens capture --camera 1` -> `python -m src.lens calibrate --square-mm 25` |
| `src/perspective.py` | Homography to a metric top-down view (ArUco IDs 0-3 at table corners) | `python -m src.perspective make-markers` |
| `src/splitter.py` | Distance-transform markers + nearest-core assignment (watershed-style) to split touching objects | `--split` flag |

Full pipeline:
```bash
python -m src.main --camera 1 --lens camera_calib.npz --warp aruco --table-mm 300 200 --split
```
Camera resolution must be identical for lens capture, calibration and the live run (default 1280x720).
