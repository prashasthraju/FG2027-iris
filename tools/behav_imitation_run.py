"""Thin wrapper: run the four behavioural embedding pipelines on outputs/behavioural/videos_imitation.json.

    python               tools/behav_imitation_run.py trustfa
    python tools/behav_imitation_run.py graph|dino|clip
Output: outputs/behavioural_imitation/<model>/emb.npz  (ids + L2-normalised emb). The original scripts are imported unchanged;
only their input list / output directory are redirected (trustfa, graph via their CLI flags; dino/clip by overriding the module-level OUTDIR).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True
VIDEOS = str(ROOT / "outputs/behavioural/videos_imitation.json")
OUT = ROOT / "outputs/behavioural_imitation"


def main():
    model, rest = sys.argv[1], sys.argv[2:]
    (OUT / model).mkdir(parents=True, exist_ok=True)
    if model == "trustfa":
        import behav_trustfa as m
        sys.argv = ["behav_trustfa", "--videos", VIDEOS, "--out", str(OUT / model)] + rest
        m.main()
    elif model == "graph":
        import behav_graph as m
        sys.argv = ["behav_graph", "--videos", VIDEOS, "--out-dir", str(OUT / model)] + rest
        m.main()
    elif model in ("dino", "clip"):
        import argparse
        import torch
        import behav_foundation as m
        m.OUTDIR = OUT                      # run() writes OUTDIR/<model>/emb.npz
        ap = argparse.ArgumentParser()
        ap.add_argument("--device", default="cuda:0")
        ap.add_argument("--bs", type=int, default=8)
        ap.add_argument("--workers", type=int, default=4)
        a = ap.parse_args(rest)
        a.model, a.videos = model, VIDEOS
        torch.set_num_threads(4)
        m.run(a)
    else:
        raise SystemExit("model must be trustfa|graph|dino|clip")


if __name__ == "__main__":
    main()
