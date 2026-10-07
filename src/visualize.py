import cv2


def draw(frame, detections, calibrator=None, ref_index=None, fps=None):
    out = frame.copy()
    for i, d in enumerate(detections):
        color = (0, 200, 255) if i == ref_index else (0, 255, 0)
        cv2.drawContours(out, [d.box], 0, color, 2)
        label = f"#{i + 1}"
        if calibrator and calibrator.is_calibrated:
            l, w = calibrator.measure(d)
            label += f" {l:.1f}x{w:.1f}mm"
        x, y = int(d.box[:, 0].min()), int(d.box[:, 1].min())
        cv2.putText(out, label, (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
    status = f"Count: {len(detections)}"
    if fps is not None:
        status += f" | FPS: {fps:.0f}"
    if not (calibrator and calibrator.is_calibrated):
        status += " | NOT CALIBRATED (press 'c')"
    cv2.putText(out, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return out
