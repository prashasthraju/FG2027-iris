"""Held-out identity judge: CVRL ArcIris (D11 P3).

Our IrisCode is *in the loop* (it guides design choices and is the binary code the
cancelable phase needs), so it cannot also be the impartial scorer: the matcher
unwraps with the same geometry we used to wrap, which flatters every result
(research/03 §0). ArcIris is a different architecture (IResNet-100 + ArcFace on
64x512 polar strips, NIR-trained, red channel, no occlusion mask, real-valued
embedding scored by angle) and ships its own circle regressor.

Two arms, deliberately:
  judge.encode_image(rgb)  - ArcIris end to end, *its own* segmentation.
                             The true held-out judge: identity AND whether our
                             edited eye is geometrically plausible to a system
                             that shares nothing with us.
  judge.encode_strip(strip) - ArcIris encoder on *our* 64x512 strip.
                             Isolates encoder from geometry in the S0 ablation.

Weights: CVRL, research use only (cite the ArcIris paper). Not redistributed;
see third_party/arciris/models/readme.txt. Code licence is CVRL's custom one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ARCIRIS_DIR = Path(__file__).resolve().parent.parent / "third_party" / "arciris"
POLAR_H, POLAR_W = 64, 512


class ArcIrisJudge:
    """Wrapper around CVRL ArcIris. Lazy-loads torch and the two networks."""

    def __init__(self, cuda=True, min_pupil_r=12, min_iris_r=16):
        if not (ARCIRIS_DIR / "models" / "ResNet100_154000.pt").exists():
            raise FileNotFoundError(
                f"ArcIris weights missing. See {ARCIRIS_DIR / 'models' / 'readme.txt'}")
        import torch
        sys.path.insert(0, str(ARCIRIS_DIR))
        from modules.irisRecognition import irisRecognition

        cuda = bool(cuda and torch.cuda.is_available())
        cfg = {
            "circle_model_path": str(ARCIRIS_DIR / "models" / "resnet18-1543-0.047488-maskIoU-0.934494.pth"),
            "nn_model_path": str(ARCIRIS_DIR / "models" / "ResNet100_154000.pt"),
            "cuda": cuda,
            "polar_height": POLAR_H, "polar_width": POLAR_W,
            "minimum_pupil_radius": min_pupil_r, "minimum_iris_radius": min_iris_r,
        }
        self._torch = torch
        self.rec = irisRecognition(cfg)
        self.cuda = cuda

    # ---------------------------------------------------------------- inputs
    @staticmethod
    def _red_u8(rgb):
        """ArcIris is trained on the RED channel of RGB (arciris.py: .split()[0])."""
        a = np.asarray(rgb)
        if a.dtype != np.uint8:
            a = np.clip(a * 255 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
        return a[..., 0] if a.ndim == 3 else a

    def encode_image(self, rgb):
        """Full ArcIris pipeline on a whole eye image: its own segmentation.
        Returns (vector, info) with vector None if its quality gate rejects."""
        from PIL import Image
        im = Image.fromarray(self._red_u8(rgb), "L")
        im = self.rec.fix_image(im)
        pupil_xyr, iris_xyr = self.rec.circApprox(im)
        if pupil_xyr is None or iris_xyr is None or not self.rec.checkQuality(pupil_xyr, iris_xyr):
            return None, dict(rejected=True, pupil=pupil_xyr, iris=iris_xyr)
        polar = self.rec.cartToPol_torch(im, pupil_xyr, iris_xyr)
        v = self.rec.extractVector(polar)
        return v, dict(rejected=False, pupil=list(map(float, pupil_xyr)),
                       iris=list(map(float, iris_xyr)), polar=polar)

    def encode_strip(self, strip):
        """ArcIris encoder on OUR normalized strip (64x512, RGB float or gray u8)."""
        import cv2
        s = np.asarray(strip)
        g = self._red_u8(s)
        if g.shape != (POLAR_H, POLAR_W):
            g = cv2.resize(g, (POLAR_W, POLAR_H), interpolation=cv2.INTER_AREA)
        return self.rec.extractVector(np.ascontiguousarray(g))

    @staticmethod
    def eye_crop(rgb, iris, width_in_radii=5.0):
        """4:3 crop centred on the iris, sized so ArcIris sees an ISO-like eye image.

        ArcIris resizes ANY input to 640x480 and regresses circles there, because
        it is trained on eye images. Feeding it a whole face shrinks the iris far
        below its quality gate: a 172 px iris in a 2352x1728 CASIA face becomes
        ~47 px, and the gate rejects it ("pupil radius too small"). With
        width = 5 r the iris lands at ~256 px diameter after the resize.
        """
        import numpy as _np
        cx, cy = float(iris[0]), float(iris[1])
        r = float(_np.sqrt(iris[2] * (iris[3] if len(iris) > 3 else iris[2])))
        w = max(width_in_radii * r, 32.0)
        h = w * 3.0 / 4.0
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        x1, y1 = int(round(cx + w / 2)), int(round(cy + h / 2))
        H, W = rgb.shape[:2]
        pad_l, pad_t = max(0, -x0), max(0, -y0)
        pad_r, pad_b = max(0, x1 - W), max(0, y1 - H)
        sub = rgb[max(y0, 0):min(y1, H), max(x0, 0):min(x1, W)]
        if pad_l or pad_t or pad_r or pad_b:
            pw = ((pad_t, pad_b), (pad_l, pad_r)) + (((0, 0),) if sub.ndim == 3 else ())
            sub = _np.pad(sub, pw, mode="edge")
        return _np.ascontiguousarray(sub)

    def circles(self, rgb):
        """ArcIris's learned pupil/iris circles, mapped back to ORIGINAL image
        pixels. D13 showed its geometry halves our EER, so this exposes it as a
        segmentation backend (`segment_closeup(..., circles=..., refine=False)`).

        fix_image pads to 4:3 with a mid-grey border and resizes to 640x480, so
        the mapping is a uniform scale plus one offset; both are inverted here.
        """
        from PIL import Image
        g = self._red_u8(rgb)
        h, w = g.shape
        ar = w / h
        if 1.333 <= ar <= 1.334:
            pad_x = pad_y = 0.0
            w_pad, h_pad = w, h
        elif ar < 1.333:
            w_pad, h_pad = h * 4.0 / 3.0, h
            pad_x, pad_y = (w_pad - w) / 2, 0.0
        else:
            w_pad, h_pad = w, w * 3.0 / 4.0
            pad_x, pad_y = 0.0, (h_pad - h) / 2
        im = self.rec.fix_image(Image.fromarray(g, "L"))
        pupil, iris = self.rec.circApprox(im)
        if pupil is None or iris is None:
            return None, None, dict(rejected=True)
        sx, sy = self.rec.ISO_RES[0] / w_pad, self.rec.ISO_RES[1] / h_pad
        s = 0.5 * (sx + sy)                     # uniform after padding

        def back(c):
            return (float(c[0] / sx - pad_x), float(c[1] / sy - pad_y), float(c[2] / s))

        pu, ir = back(pupil), back(iris)
        return pu, ir, dict(rejected=not self.rec.checkQuality(pupil, iris))

    # ---------------------------------------------------------------- scoring
    @staticmethod
    def distance(v1, v2):
        """ArcIris score: angular distance acos(cosine), in [0, pi]. Lower = same eye."""
        if v1 is None or v2 is None:
            return np.nan
        c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
        return float(np.arccos(np.clip(c, -1.0, 1.0)))

    def encode_strip_rotations(self, strip, shifts=range(-16, 17, 4)):
        """ArcIris has no rotation search (matchVectors is a single comparison).
        Encode the strip at several theta shifts so the caller can take the min
        distance, mirroring our IrisCode bit-shifting (research/01 Step 1.4)."""
        import cv2
        s = np.asarray(strip)
        g = self._red_u8(s)
        if g.shape != (POLAR_H, POLAR_W):
            g = cv2.resize(g, (POLAR_W, POLAR_H), interpolation=cv2.INTER_AREA)
        return [self.rec.extractVector(np.ascontiguousarray(np.roll(g, k, axis=1))) for k in shifts]

    def min_distance(self, probe_rotations, gallery_vector):
        return min((self.distance(v, gallery_vector) for v in probe_rotations), default=np.nan)
