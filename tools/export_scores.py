"""Export per-session scalar scores so the headline numbers can be recomputed without any licensed data.

    python tools/export_scores.py       # -> outputs/scores_liveportrait.json

For every LivePortrait honest session (tuning and test) it stores the mean ArcIris angular distance from the video to each enrolled
proxy (one float per proxy, or null when the video has fewer than 5 usable iris observations), and the index of the driver's proxy.
No embeddings, images or templates are written: the numbers are scalars derived from the extraction caches and the issuer's registry.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import proxy_ledger as PL  # noqa: E402
import proxy_verify as V  # noqa: E402
from provenance import stamp  # noqa: E402


def main():
    plan = json.loads(V.PLAN.read_text())
    enrol = plan["enrolment"]
    G = V.Gallery(PL.Ledger.load(V.LEDGER))
    didx = {n[0]: i for i, n in enumerate(G.names)}
    sessions = []
    for m in V.sessions():
        if m["kind"] not in ("self", "cross"):
            continue
        c = V.load_cache(m["id"])
        if c is None:
            continue
        d = G.vectors(c)[0]["arciris"]
        sessions.append(dict(id=m["id"], kind=m["kind"], dataset=m["dataset"], tuning=bool(m["tuning"]),
                             driver_proxy=didx[enrol[m["driver_actor"]]["did"]],
                             distance=[round(float(x), 8) if np.isfinite(x) else None for x in d]))
    out = dict(note="mean ArcIris angular distance (rad) of the video to every enrolled proxy; null = fewer than 5 usable iris observations",
               proxies=[n[0] for n in G.names], sessions=sessions)
    (ROOT / "outputs" / "scores_liveportrait.json").write_text(json.dumps(stamp(out), separators=(",", ":")))
    print(len(sessions), "sessions,", len(G.names), "proxies")


if __name__ == "__main__":
    main()
