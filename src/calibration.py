"""Scale calibration using a reference object of known real-world size."""
import json
from pathlib import Path


class Calibrator:
    def __init__(self, pixels_per_mm=None):
        self.pixels_per_mm = pixels_per_mm

    @property
    def is_calibrated(self):
        return self.pixels_per_mm is not None

    def calibrate(self, ref_detection, ref_long_mm):
        """Use the reference object's LONG side (orientation independent)."""
        if ref_long_mm <= 0:
            raise ValueError("Reference size must be positive")
        self.pixels_per_mm = ref_detection.long_px / ref_long_mm
        return self.pixels_per_mm

    def to_mm(self, pixels):
        if not self.is_calibrated:
            raise RuntimeError("Calibrate first")
        return pixels / self.pixels_per_mm

    def measure(self, det):
        """Return (length_mm, width_mm)."""
        return self.to_mm(det.long_px), self.to_mm(det.short_px)

    def save(self, path="calibration.json"):
        Path(path).write_text(json.dumps({"pixels_per_mm": self.pixels_per_mm}))

    @classmethod
    def load(cls, path="calibration.json"):
        return cls(json.loads(Path(path).read_text())["pixels_per_mm"])
