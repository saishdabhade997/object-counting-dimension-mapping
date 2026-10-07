"""Split touching / overlapping objects (distance-transform markers + nearest-core region growing).

Same idea as the classic distance-transform watershed: (1) the distance transform's peaks
give one 'core' per object, (2) every remaining pixel is assigned to its nearest core.
cv2.watershed floods by image *gradient*, which gives uneven cuts on a flat distance map,
so the flooding step is done explicitly (per-core distance transforms + argmin).
"""
import cv2
import numpy as np


def split_touching(contour, peak_ratio=0.7, min_peak_px=3):
    """Return [contour] for a single object, or one contour per object if it is a merged blob.

    peak_ratio: a pixel belongs to an object core if its distance-to-background exceeds
    peak_ratio * (component's max distance). Higher = splits more aggressively.
    """
    x, y, w, h = cv2.boundingRect(contour)
    pad = 2
    off = np.array([[x - pad, y - pad]]).reshape(1, 1, 2)
    comp = np.zeros((h + 2 * pad, w + 2 * pad), np.uint8)
    cv2.drawContours(comp, [contour - off], -1, 255, -1)

    dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
    if dist.max() < min_peak_px:
        return [contour]
    cores = (dist > peak_ratio * dist.max()).astype(np.uint8)
    n, labels = cv2.connectedComponents(cores)
    if n <= 2:                      # background + 1 core -> single object
        return [contour]

    # distance of every pixel to each core, then assign to the nearest one
    d_to_core = [cv2.distanceTransform((labels != k).astype(np.uint8), cv2.DIST_L2, 5)
                 for k in range(1, n)]
    owner = np.argmin(np.stack(d_to_core), axis=0)

    out = []
    for k in range(n - 1):
        region = ((owner == k) & (comp > 0)).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            out.append(max(cnts, key=cv2.contourArea) + off)
    return out or [contour]
