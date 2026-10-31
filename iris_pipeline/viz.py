"""Debug visualisations."""
import cv2
import numpy as np


def eye_panel(rgb, seg, size=240):
    """Zoomed eye crop with pupil (blue), limbus (green), lids (red), visible mask tint."""
    x0, y0, x1, y1 = seg.bbox
    crop = rgb[y0:y1, x0:x1].copy()
    tint = crop.copy()
    tint[seg.visible] = (0.6 * tint[seg.visible] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)
    tint[seg.specular] = (255, 0, 255)
    k = size / max(crop.shape[:2])
    big = cv2.resize(np.hstack([crop, tint]), None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
    big = np.ascontiguousarray(big[:, :, ::-1])
    half = big.shape[1] // 2
    halves = []
    for h in (big[:, :half].copy(), big[:, half:].copy()):
        for c, col in ((seg.pupil, (255, 0, 0)), (seg.iris, (0, 255, 0))):
            cv2.circle(h, (int((c[0] - x0) * k), int((c[1] - y0) * k)), int(c[2] * k), col, 1, cv2.LINE_AA)
        cv2.polylines(h, [np.int32((seg.lid_polygon - [x0, y0]) * k)], True, (0, 0, 255), 1, cv2.LINE_AA)
        halves.append(h)
    return np.hstack(halves)  # BGR


def strip_panel(strip, mask, width=512):
    s = np.uint8(np.clip(strip, 0, 1) * 255)[:, :, ::-1].copy()
    s[~mask] = (0.35 * s[~mask]).astype(np.uint8)
    return cv2.resize(s, (width, int(width * s.shape[0] / s.shape[1])), interpolation=cv2.INTER_NEAREST)


def stack(panels, pad=4):
    w = max(p.shape[1] for p in panels)
    rows = [cv2.copyMakeBorder(p, pad, 0, 0, w - p.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30)) for p in panels]
    return np.vstack(rows)
