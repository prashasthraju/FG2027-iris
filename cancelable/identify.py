# LEGACY: part of an earlier design (public proxy codes / 1:N identification of the proxy in the video / local Anvil chain). The submitted protocol
# (commit-and-reveal, 1:1 check against the opened proxy) is implemented in tools/proxy_issuer_eval.py (verify_session) and tools/proxy_ledger.py; see README.md.
"""Which pool entry does this face (or clip) carry?

Every frame is re-detected from scratch; each eye is cropped for ArcIris
(`judge.eye_crop`, D17) and compared with the pool gallery of the *same* image side.
Per entry, distances are fused by the mean over frames and both eyes.

Plain rank-1 always names some entry, so an un-embedded or foreign video would still
"match" someone. The decision therefore needs a rejection rule: the best entry is
accepted only if the fused distance ratio best / second-best is below `ratio_max`,
calibrated on un-embedded controls (`tools/cancelable_demo.py calibrate`). The ratio
is used rather than a raw distance because ArcIris distances are not normalised per
probe (D17 caveat).
"""
import numpy as np

from iris_pipeline.segment import extract_irises

RATIO_MAX = 0.80        # replaced by the calibrated value in chain/state/calibration.json


# Geometry agreement (README.md). `eye_crop` is centred on MediaPipe's iris and sized so that iris has
# radius 128 px in ArcIris's 640x480 frame; ArcIris regresses its own circles. On soft rendered eyes its circle disagrees
# with MediaPipe's on about half of the frames, and only the agreeing frames carry the pool iris (per-frame top-1 of the
# true entry: 0.56 when they agree, 0.13 when not). Frames are therefore gated, not repaired (a MediaPipe circle with a
# fixed pupil ratio gave no gain, D16). Tolerances were fixed before scoring.
GEO_TOL_C, GEO_TOL_R, GEO_MIN_OBS = 0.15, 0.15, 4
ISO_CX, ISO_CY, ISO_R = 320.0, 240.0, 128.0


def geometry_agrees(iris_xyr):
    off = np.hypot(iris_xyr[0] - ISO_CX, iris_xyr[1] - ISO_CY) / ISO_R
    return bool(off < GEO_TOL_C and abs(iris_xyr[2] / ISO_R - 1.0) < GEO_TOL_R)


def frame_observations(rgb, det, judge, gallery):
    """[dict(side, best={sid: min ArcIris distance}, geo_ok)] for one frame (eyes ArcIris rejected are dropped)."""
    out = []
    for seg in extract_irises(rgb, det):
        if seg.side not in ("image_left", "image_right"):
            continue
        v, info = judge.encode_image(judge.eye_crop(rgb, seg.iris))
        if info["rejected"]:
            continue
        best = {}
        for sid, side, g in gallery:
            if side != seg.side:
                continue
            d = judge.distance(v, g)
            if not np.isnan(d) and d < best.get(sid, np.inf):
                best[sid] = d
        out.append(dict(side=seg.side, best=best, geo_ok=geometry_agrees(info["iris"])))
    return out


def frame_distances(rgb, det, judge, gallery):
    """{side: {sid: min ArcIris distance}} for one frame."""
    return {o["side"]: o["best"] for o in frame_observations(rgb, det, judge, gallery)}


def _rowz(row):
    """Z of one observation's distances over the entry crowd (median / 1.4826 MAD), so frames with different probe offsets
    (ArcIris distances are not normalised per probe, D17) are comparable."""
    med = np.nanmedian(row)
    return (row - med) / (1.4826 * np.nanmedian(np.abs(row - med)) + 1e-9)


def entry_scores(frames, det, judge, gallery, ids, gate=True):
    """Per-entry clip score over `ids` (lower = closer), the same quantity the cache study scores.
    gate=False: mean distance over frames and eyes (the original rule). gate=True: mean over the observations whose
    ArcIris circle agrees with MediaPipe's, each first Z-normalised over the crowd; falls back to all observations when
    fewer than GEO_MIN_OBS agree."""
    obs = [o for rgb in frames for o in frame_observations(rgb, det, judge, gallery)]
    if not obs:
        return [float("nan")] * len(ids), 0
    rows = np.array([[o["best"].get(s, np.nan) for s in ids] for o in obs])
    if gate:
        ok = np.array([o["geo_ok"] for o in obs])
        rows = rows[ok] if ok.sum() >= GEO_MIN_OBS else rows
        z = np.array([_rowz(r) for r in rows])
        return [float(x) for x in np.nanmean(z, 0)], int(len(rows))
    return [float(x) for x in np.nanmean(rows, 0)], int(len(rows))


FUSE = {"mean": np.mean, "q25": lambda x: np.percentile(x, 25),
        "q10": lambda x: np.percentile(x, 10), "min": np.min}


def identify(frames, det, judge, gallery, ratio_max=RATIO_MAX, fuse="mean", gate=False):
    """Fused decision over frames and eyes -> dict(sid or None, ratio, ranking, n_eye_obs).

    fuse: per-entry statistic over (frame, eye) distances. "mean" suits stills; on
    reenacted video, where some frames are badly segmented (soft rendered eyes) and
    give near-random distances, a low quantile keeps the frames where the inserted
    iris was recovered.
    gate: keep only the observations whose ArcIris circle agrees with MediaPipe's (D40; falls back to all when fewer than
    GEO_MIN_OBS agree). Off by default so stored results reproduce; on, the recalculated held-out rule gives 41/45, 42/45 and
    18/27 at the same tau (docs/12)."""
    obs = [o for rgb in frames for o in frame_observations(rgb, det, judge, gallery)]
    if gate and sum(o["geo_ok"] for o in obs) >= GEO_MIN_OBS:
        obs = [o for o in obs if o["geo_ok"]]
    acc, n = {}, 0
    for o in obs:
        n += 1
        for sid, d in o["best"].items():
            acc.setdefault(sid, []).append(d)
    if not acc:
        return dict(sid=None, reason="no eye measured", ratio=None, ranking=[], n_eye_obs=0)
    fused = sorted(((float(FUSE[fuse](d)), sid) for sid, d in acc.items()))
    ratio = fused[0][0] / fused[1][0] if len(fused) > 1 else 0.0
    ok = ratio < ratio_max
    return dict(sid=fused[0][1] if ok else None, reason=None if ok else f"ambiguous (ratio {ratio:.3f} >= {ratio_max})",
                ratio=round(ratio, 4), ranking=[(sid, round(d, 4)) for d, sid in fused[:5]], n_eye_obs=n)
