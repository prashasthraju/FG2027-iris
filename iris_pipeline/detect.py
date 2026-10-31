"""Stage 1 - coarse eye/iris localisation with MediaPipe Face Landmarker.

The 478-point face mesh includes 10 iris landmarks (5 per eye: centre + 4 on
the limbus) and dense eyelid contours. We use them as a *prior*; the circles
are refined photometrically in segment.py.
"""
from __future__ import annotations

import atexit
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "face_landmarker.task"

# Eyelid contours (closed loops) from the canonical MediaPipe face mesh.
EYE_CONTOURS = {
    "image_left": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    "image_right": [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
}
IRIS_GROUPS = [list(range(468, 473)), list(range(473, 478))]  # [centre, 4 limbus points]


@dataclass
class EyeLandmarks:
    side: str                 # "image_left" / "image_right" (as seen in the picture)
    iris_center: np.ndarray   # (2,) px
    iris_radius: float        # px, from the 4 limbus landmarks
    lid_polygon: np.ndarray   # (16, 2) px, closed eyelid contour


class FaceEyeDetector:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL):
        import mediapipe as mp

        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.CPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
        )
        self._mp = mp
        self._lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        # MediaPipe dispatches close() through a thread pool that is shut down
        # before regular atexit hooks run; threading's exit hooks run earlier
        # (reverse registration order), so close there to avoid a __del__ traceback.
        getattr(threading, "_register_atexit", atexit.register)(self.close)

    def detect(self, rgb: np.ndarray) -> list[EyeLandmarks]:
        h, w = rgb.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        res = self._lm.detect(img)
        if not res.face_landmarks:
            return []
        pts = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float64)
        if len(pts) < 478:
            raise RuntimeError("Face landmarker model has no iris landmarks (need 478 points)")

        eyes = []
        for side, idx in EYE_CONTOURS.items():
            lid = pts[idx]
            lid_c = lid.mean(0)
            # pair each eye contour with the nearest iris group (index order is model-specific)
            grp = min(IRIS_GROUPS, key=lambda g: np.linalg.norm(pts[g[0]] - lid_c))
            centre, limbus = pts[grp[0]], pts[grp[1:]]
            radius = float(np.linalg.norm(limbus - centre, axis=1).mean())
            eyes.append(EyeLandmarks(side, centre, radius, lid))
        return eyes

    # Use as a context manager: closing before interpreter teardown avoids
    # MediaPipe's noisy __del__ TypeError (atexit is too late for its executor).
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self._lm is not None:
            self._lm.close()
            self._lm = None
