"""Prepare a colour, consented-people target set from the RAVDESS video-speech zips (README.md).

    python tools/ravdess_prep.py [--actors 01 02 ...] [--no-download]

RAVDESS (Livingstone & Russo 2018, Zenodo 1188976, CC BY-NC-SA 4.0) is licensed data: everything lands in
dbs/ravdess_video/ (git-ignored, local only). Per actor, resumable (a finished actor is skipped, --force redoes):

  1. download Video_Speech_Actor_NN.zip (curl -C - resume), check the Zenodo md5 and `unzip -t`;
  2. extract the audio-video clips (modality 01, the ones in ravdess_metadata.csv);
  3. SOURCE  = neutral / normal / statement 1 / repetition 1 clip: the sharpest frame in which both eyes are open,
               frontal and pass iris_pipeline.quality (size excepted: RAVDESS irises are ~29 px, always < 60 px),
               saved as a 512x512 face crop (no resampling, so the iris keeps its native size);
  4. DRIVERS = up to 2 clips of statement 2, calm / happy, normal intensity (strong as fallback), eyes open in
               >= 60 % of the cut; cut to 120 frames (all frames if the clip is shorter) as a face-crop mp4;
  5. outputs/ravdess_targets_index.json (provenance-stamped) lists actor, source png, drivers, iris px, emotion, statement.

Native iris diameters are measured with the project's own segmenter (MediaPipe circle at this size).
"""
import os
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from iris_pipeline import quality  # noqa: E402
from iris_pipeline.detect import FaceEyeDetector  # noqa: E402
from iris_pipeline.segment import extract_irises  # noqa: E402
from provenance import stamp  # noqa: E402

DATA = ROOT / "dbs" / "ravdess_video"
META = Path(os.environ.get("RAVDESS_METADATA", "data/metadata/ravdess_metadata.csv"))
INDEX = ROOT / "outputs" / "ravdess_targets_index.json"
RECORD = DATA / "zenodo_record_1188976.json"
ZENODO = "https://zenodo.org/records/1188976/files/{}?download=1"
DEFAULT_ACTORS = ["01", "02", "03", "05", "08", "11", "13", "16", "20", "22"]
OPEN_APERTURE = 0.20          # lid-polygon height / width below this = lids (nearly) closed
FRONTAL_WIDTH_RATIO = (0.80, 1.25)   # left/right eye-opening width: head yaw proxy
CROP = 512
DRIVER_FRAMES = 120
MIN_OPEN_FRAC = 0.60


# ---------------------------------------------------------------- download / verify
def md5_of(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ensure_zip(actor, download=True):
    name = f"Video_Speech_Actor_{actor}.zip"
    zp, ok = DATA / name, DATA / f".{name}.verified"
    if ok.exists() and zp.exists():
        return zp
    rec = json.loads(RECORD.read_text())
    meta = next(f for f in rec["files"] if f["key"] == name)
    if download and (not zp.exists() or zp.stat().st_size != meta["size"]):
        subprocess.run(["curl", "-L", "--fail", "-C", "-", "--retry", "5", "--retry-delay", "5", "-sS",
                        "-o", str(zp), ZENODO.format(name)], check=True)
    if not zp.exists() or zp.stat().st_size != meta["size"]:
        raise SystemExit(f"{name}: incomplete ({zp.stat().st_size if zp.exists() else 0}/{meta['size']} B)")
    want = meta["checksum"].split(":", 1)[1]
    got = md5_of(zp)
    if got != want:
        raise SystemExit(f"{name}: md5 {got} != Zenodo {want}")
    subprocess.run(["unzip", "-tq", str(zp)], check=True)
    ok.write_text(f"md5 {got}\n")
    return zp


def extract_av(actor, zp, names):
    d = DATA / f"Actor_{actor}"
    d.mkdir(exist_ok=True)
    with zipfile.ZipFile(zp) as z:
        by_base = {Path(n).name: n for n in z.namelist() if n.endswith(".mp4")}
        for n in names:
            dst = d / n
            if not dst.exists():
                dst.write_bytes(z.read(by_base[n]))
    return d


# ---------------------------------------------------------------- frame analysis
def read_frames(path):
    cap, out = cv2.VideoCapture(str(path)), []
    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return out, fps


def analyse(rgb, det):
    """Both eyes of one frame -> dict, or None when MediaPipe does not return exactly two eyes."""
    segs = extract_irises(rgb, det)
    if len(segs) != 2:
        return None
    eyes = []
    for s in segs:
        v = quality.assess(s, "target")
        # the size reason ("iris 29 px (< 60 ...") is expected here and not a reason to reject
        reasons = [r for r in v.reasons if "px (<" not in r]
        poly = np.asarray(s.lid_polygon, float)
        w, h = np.ptp(poly[:, 0]), np.ptp(poly[:, 1])
        eyes.append(dict(side=s.side, diameter_px=float(s.diameter_px), width=float(w), aperture=float(h / max(w, 1e-6)),
                         gated=bool(reasons), reasons=reasons, centre=[float(s.iris[0]), float(s.iris[1])],
                         poly=poly, scores=v.scores))
    ratio = eyes[0]["width"] / max(eyes[1]["width"], 1e-6)
    open_ = all(e["aperture"] >= OPEN_APERTURE and not e["gated"] for e in eyes)
    frontal = FRONTAL_WIDTH_RATIO[0] <= ratio <= FRONTAL_WIDTH_RATIO[1]
    return dict(eyes=eyes, open=open_, frontal=frontal, width_ratio=float(ratio),
                diameter_px=float(np.mean([e["diameter_px"] for e in eyes])))


def eye_sharpness(rgb, info):
    """Variance of the Laplacian over the eye band (both eyes + margin)."""
    pts = np.vstack([e["poly"] for e in info["eyes"]])
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    m = 0.5 * (x1 - x0) * 0.25
    h, w = rgb.shape[:2]
    x0, x1 = int(max(x0 - m, 0)), int(min(x1 + m, w))
    y0, y1 = int(max(y0 - 3 * m, 0)), int(min(y1 + 3 * m, h))
    g = cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def crop_box(infos, shape):
    """Fixed CROPxCROP box around the median eye midpoint (face sits a little below the eyes' line)."""
    mid = np.median([np.mean([e["centre"] for e in i["eyes"]], 0) for i in infos], 0)
    h, w = shape[:2]
    x0 = int(np.clip(mid[0] - CROP / 2, 0, max(w - CROP, 0)))
    y0 = int(np.clip(mid[1] - 0.40 * CROP, 0, max(h - CROP, 0)))
    return x0, y0, min(CROP, w), min(CROP, h)


def write_mp4(frames, path, fps):
    h, w = frames[0].shape[:2]
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                          "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "12", "-preset", "veryfast",
                          "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    if p.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")


# ---------------------------------------------------------------- selection
def pick_source(path, det):
    frames, _ = read_frames(path)
    best, n_ok, cands = None, 0, []
    for i, f in enumerate(frames):
        info = analyse(f, det)
        if info is None or not (info["open"] and info["frontal"]):
            continue
        n_ok += 1
        sh = eye_sharpness(f, info)
        cands.append(info["diameter_px"])
        if best is None or sh > best[0]:
            best = (sh, i, info, f)
    if best is None:
        return None
    sh, i, info, f = best
    x0, y0, w, h = crop_box([info], f.shape)
    return dict(frame=i, n_frames=len(frames), n_candidate_frames=n_ok, sharpness=sh, crop=[x0, y0, w, h],
                iris_px=info["diameter_px"], iris_px_per_eye={e["side"]: e["diameter_px"] for e in info["eyes"]},
                iris_px_median_candidates=float(np.median(cands)), min_aperture=min(e["aperture"] for e in info["eyes"]),
                width_ratio=info["width_ratio"], image=f[y0:y0 + h, x0:x0 + w])


def pick_driver(path, det):
    frames, fps = read_frames(path)
    frames = frames[:DRIVER_FRAMES]
    infos = [analyse(f, det) for f in frames]
    n_open = sum(1 for i in infos if i is not None and i["open"])
    good = [i for i in infos if i is not None]
    if not good:
        return None
    x0, y0, w, h = crop_box(good, frames[0].shape)
    return dict(n_frames=len(frames), fps=float(fps), open_frac=n_open / len(frames), crop=[x0, y0, w, h],
                iris_px_median=float(np.median([i["diameter_px"] for i in good])),
                frames=[f[y0:y0 + h, x0:x0 + w] for f in frames])


def rows_for(actor):
    with open(META) as fh:
        return [r for r in csv.DictReader(fh) if r["actor_id"] == f"Actor_{actor}" and r["modality"] == "AV"]


def prep_actor(actor, det, download, force):
    rows = rows_for(actor)
    base = lambda r: Path(r["filepath"]).name
    src_row = next(r for r in rows if r["emotion_code"] == "1" and r["intensity_code"] == "1"
                   and r["statement_code"] == "1" and r["repetition"] == "1")
    drv_rows = sorted((r for r in rows if r["emotion_code"] in ("2", "3") and r["statement_code"] == "2"),
                      key=lambda r: (r["intensity_code"] != "1", r["emotion_code"], r["repetition"]))
    zp = ensure_zip(actor, download)
    d = extract_av(actor, zp, [base(src_row)] + [base(r) for r in drv_rows])
    out = DATA / f"Actor_{actor}"
    res = dict(actor=f"Actor_{actor}", gender=src_row["actor_gender"], age=src_row["actor_age"],
               ethnicity=src_row["actor_ethnicity"], camera=src_row["camera_angle"],
               native_resolution=f"{src_row['resolution_w']}x{src_row['resolution_h']}", zip=zp.name)

    s = pick_source(d / base(src_row), det)
    if s is None:
        res["error"] = "no frontal both-eyes-open frame in the neutral clip"
        return res
    png = out / f"source_{Path(base(src_row)).stem}_f{s['frame']:03d}.png"
    cv2.imwrite(str(png), cv2.cvtColor(s.pop("image"), cv2.COLOR_RGB2BGR))
    res["source"] = dict(png=str(png.relative_to(ROOT)), clip=src_row["filepath"], emotion=src_row["emotion"],
                         intensity=src_row["intensity"], statement=src_row["statement"], repetition=int(src_row["repetition"]),
                         **s)
    res["iris_px"] = s["iris_px"]

    res["drivers"] = []
    # best normal-intensity clip of each emotion first (calm, happy), then whatever else is left
    seen_emotions, order = set(), []
    for r in drv_rows:
        if r["intensity_code"] == "1" and r["emotion"] not in seen_emotions:
            seen_emotions.add(r["emotion"])
            order.append(r)
    order += [r for r in drv_rows if r not in order]
    tried = []
    for r in order:
        if len(res["drivers"]) >= 2:
            break
        dv = pick_driver(d / base(r), det)
        tried.append(dict(clip=r["filepath"], open_frac=None if dv is None else round(dv["open_frac"], 3)))
        if dv is None or dv["open_frac"] < MIN_OPEN_FRAC:
            continue
        mp4 = out / f"driver_{Path(base(r)).stem}.mp4"
        write_mp4(dv.pop("frames"), mp4, dv["fps"])
        res["drivers"].append(dict(mp4=str(mp4.relative_to(ROOT)), clip=r["filepath"], emotion=r["emotion"],
                                   intensity=r["intensity"], statement=r["statement"], repetition=int(r["repetition"]), **dv))
    res["drivers_tried"] = tried
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", nargs="*", default=DEFAULT_ACTORS)
    ap.add_argument("--no-download", action="store_true", help="only process zips that are already complete")
    ap.add_argument("--force", action="store_true", help="redo actors already in the index")
    a = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    INDEX.parent.mkdir(exist_ok=True)
    old = json.loads(INDEX.read_text()) if INDEX.exists() else {}
    done = {t["actor"]: t for t in old.get("targets", [])}
    with FaceEyeDetector() as det:
        for actor in a.actors:
            key = f"Actor_{actor}"
            t = done.get(key)
            if t and not a.force and "error" not in t and (ROOT / t["source"]["png"]).exists() and \
                    all((ROOT / d["mp4"]).exists() for d in t["drivers"]):
                print(f"{key}: already prepared, iris {t['iris_px']:.1f} px")
                continue
            try:
                t = prep_actor(actor, det, not a.no_download, a.force)
            except SystemExit as e:
                print(f"{key}: SKIPPED ({e})")
                continue
            done[key] = t
            print(f"{key}: iris {t.get('iris_px', float('nan')):.1f} px, {len(t.get('drivers', []))} drivers"
                  f"{'  ERROR ' + t['error'] if 'error' in t else ''}")
            INDEX.write_text(json.dumps(stamp(dict(
                dataset="RAVDESS video-speech (Livingstone & Russo 2018, Zenodo 1188976, CC BY-NC-SA 4.0): local research use only",
                note="iris_px = native iris diameter in the 1280x720 video (MediaPipe circle), mean of both eyes; "
                     "images and clips are licensed data under dbs/ravdess_video/ and are not to be published",
                targets=[done[k] for k in sorted(done)]), seed=None), indent=1))


if __name__ == "__main__":
    main()
