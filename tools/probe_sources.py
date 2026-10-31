"""Probe iris size / eye openness across the video sources and keep the
widest-open frame per video (frame picker)."""
import glob
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from iris_pipeline.detect import FaceEyeDetector

OUT = Path(__file__).resolve().parent.parent / "samples" / "probe"


def openness(e):
    p = e.lid_polygon
    return (p[:, 1].max() - p[:, 1].min()) / (p[:, 0].max() - p[:, 0].min() + 1e-9)


def best_frame(det, video, n_samples=8):
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best = None
    for t in np.linspace(0, n - 1, n_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, f = cap.read()
        if not ok:
            continue
        eyes = det.detect(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if len(eyes) < 2:
            continue
        o = min(openness(e) for e in eyes)
        if best is None or o > best[0]:
            best = (o, f, eyes)
    return best


def main(per_source=6):
    OUT.mkdir(parents=True, exist_ok=True)
    det = FaceEyeDetector()
    random.seed(0)
    for src in ["TEST/LIVE/Actor_*", "TEST/LIVE/C1*", "TEST/GAGA/*"]:
        vids = sorted(glob.glob(f"../data/videos/{src}.mp4"))
        random.shuffle(vids)
        diam, shapes = [], set()
        for v in vids[:per_source]:
            b = best_frame(det, v)
            if b is None:
                continue
            o, f, eyes = b
            shapes.add(f.shape[:2])
            diam.append(2 * np.mean([e.iris_radius for e in eyes]))
            name = Path(v).name.split("--")[0] + "_" + Path(v).parent.name
            cv2.imwrite(str(OUT / f"{name}.png"), f)
        print(f"{src:20s} {shapes} iris diameter px mean {np.mean(diam):.1f} "
              f"min {np.min(diam):.1f} max {np.max(diam):.1f} (n={len(diam)})")


if __name__ == "__main__":
    main()
