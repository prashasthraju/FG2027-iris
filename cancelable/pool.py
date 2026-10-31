"""The random iris pool: the cancelable layer's identities (SYNTHETIC_PLAN option A, D24).

A pool entry is one UTIRIS subject: its right-eye and left-eye capture-1 strips are
what gets embedded, and its captures 2..5 are the verification gallery, so a match
is never against the pixels that were embedded. The pool iris is unrelated to
the person it is assigned to, so it reveals nothing about their real iris (R3) and a
new key simply gets a new entry (R1). The synthetic generator replaces this pool
later through the same 64x512 strip interface.

Eye placement: a person's right eye appears on the image left, so the entry's R strip
goes to the "image_left" eye and its L strip to "image_right".
"""
import json
from pathlib import Path

import cv2
import numpy as np

from iris_pipeline.embed import IrisTexture
from iris_pipeline.segment import segment_closeup

ROOT = Path(__file__).resolve().parent.parent
VIS_ROOT = ROOT / "dbs" / "UTIRIS_V.1" / "RGB Images"
CACHE = ROOT / "outputs" / "pool"
SIDE_OF = {"R": "image_left", "L": "image_right"}


def _load_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def scan():
    """sid -> {eye -> {capture -> path}} for subjects usable as pool entries:
    both eyes, capture 1 present, and at least one other capture per eye."""
    by = {}
    for p in sorted(VIS_ROOT.rglob("*")):
        if p.suffix.lower() != ".jpg" or p.name.startswith("._"):
            continue
        _, sid, eye, cap = p.stem.split("_")
        by.setdefault(sid, {}).setdefault(eye, {})[int(cap)] = p
    return {sid: e for sid, e in by.items()
            if all(eye in e and 1 in e[eye] and len(e[eye]) > 1 for eye in ("L", "R"))}


class IrisPool:
    def __init__(self):
        self.files = scan()
        self.ids = sorted(self.files)

    def __len__(self):
        return len(self.ids)

    def textures(self, sid):
        """{"image_left": IrisTexture(R capture 1), "image_right": IrisTexture(L capture 1)},
        segmented once and cached under outputs/pool/ (licensed data, git-ignored)."""
        out = {}
        for eye, side in SIDE_OF.items():
            f = CACHE / f"{sid}_{eye}.npz"
            if f.exists():
                d = np.load(f)
                out[side] = IrisTexture(d["strip"], d["mask"], f"pool:{sid}_{eye}")
                continue
            seg = segment_closeup(_load_rgb(self.files[sid][eye][1]))
            CACHE.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, strip=seg.strip, mask=seg.strip_mask)
            out[side] = IrisTexture(seg.strip, seg.strip_mask, f"pool:{sid}_{eye}")
        return out

    def save_strips(self, sid, out_dir):
        """PNG strips in video.py's order (image-left eye first)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tex, paths = self.textures(sid), []
        for side in ("image_left", "image_right"):
            p = out_dir / f"pool_{sid}_{side}.png"
            cv2.imwrite(str(p), cv2.cvtColor(np.uint8(np.clip(tex[side].strip, 0, 1) * 255 + 0.5), cv2.COLOR_RGB2BGR))
            paths.append(p)
        return paths

    def gallery(self, judge, cache=CACHE / "gallery_arciris.npz"):
        """ArcIris vectors of captures 2..5 of every entry, labelled (sid, image side)."""
        if cache.exists():
            d = np.load(cache, allow_pickle=True)
            return [tuple(x) for x in d["items"]]
        items = []
        for sid in self.ids:
            for eye, side in SIDE_OF.items():
                for cap, p in sorted(self.files[sid][eye].items()):
                    if cap == 1:
                        continue
                    v, info = judge.encode_image(_load_rgb(p))
                    if not info["rejected"]:
                        items.append((sid, side, v))
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, items=np.array(items, dtype=object))
        return items
