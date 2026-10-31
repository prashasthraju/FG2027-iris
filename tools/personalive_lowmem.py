"""Run PersonaLive's own inference_offline.py with lower GPU memory, without editing its repo.

PersonaLive denoises temporal_window_size x temporal_adaptive_step frames per UNet pass
(4 x 4 = 16 frames by default), which does not fit an 8 GB GPU next to its three UNets.
This wrapper patches the defaults of Pose2VideoPipeline_Stream.__call__ and then runs the
unchanged script. Options, cheapest first:

  --tas N        temporal_adaptive_step (must divide the 4 denoising steps: 1, 2, 4)
  --tws N        temporal_window_size (must divide the clip length)
  --vae-slicing  decode one frame at a time (same pixels, lower peak memory)
  --embed-entry SID   inline cancelable iris: every decoded frame is sent to
                 tools/embed_worker.py (conda env) the moment PersonaLive generates it, upscaled there
                 so the iris reaches --target-iris-px;
                 raw and embedded frames are written losslessly to --out
  --int8         quantize the denoising UNet's Linear layers to 8-bit (bitsandbytes LLM.int8)

Everything after "--" goes to inference_offline.py unchanged, e.g.

    python -s tools/personalive_lowmem.py --tas 2 -- \\
        --config configs/personalive_rebuilt.yaml --reference_image R.png --driving_video D.mp4 -L 16
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

PL = Path(os.environ.get("PERSONALIVE_DIR", "external/PersonaLive"))


def main():
    argv = sys.argv[1:]
    rest = argv[argv.index("--") + 1:] if "--" in argv else []
    ours = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--tas", type=int, default=4)
    ap.add_argument("--tws", type=int, default=4)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--vae-slicing", action="store_true", help="decode one frame at a time (same output)")
    ap.add_argument("--embed-entry", default="")
    ap.add_argument("--target-iris-px", type=float, default=150,
                    help="the embedder upscales frames so the iris reaches this size")
    ap.add_argument("--out", default="")
    a = ap.parse_args(ours)
    rest = [str(Path(x).resolve()) if x.endswith((".yaml", ".png", ".jpg", ".mp4")) and Path(x).exists() else x
            for x in rest]

    out_dir = Path(a.out).resolve() if a.out else None
    os.chdir(PL)
    sys.path.insert(0, str(PL))
    from src.pipelines import pipeline_pose2vid as pp

    raw, embedded = [], []
    if a.embed_entry:
        import cv2
        import numpy as np
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from cancelable.inline import WorkerEmbedder     # PersonaLive's env -> conda-env worker
        embedder = WorkerEmbedder(a.embed_entry, target_iris_px=a.target_iris_px)
        decode = pp.Pose2VideoPipeline_Stream.decode_latents

        def decode_and_embed(self, latents, *x, **k):      # adapter: PersonaLive's per-window decode
            video = decode(self, latents, *x, **k)          # (b, c, f, h, w) in [0, 1]: one window
            for f in range(video.shape[2]):                 # each frame, as soon as it exists
                fr = (video[0, :, f].permute(1, 2, 0).numpy() * 255 + 0.5).astype(np.uint8)
                out = embedder(fr)
                raw.append(cv2.resize(fr, out.shape[1::-1], interpolation=cv2.INTER_CUBIC))
                embedded.append(out)
            return video
        pp.Pose2VideoPipeline_Stream.decode_latents = decode_and_embed

    call = pp.Pose2VideoPipeline_Stream.__call__

    def lowmem_call(self, *args, **kw):
        kw["temporal_adaptive_step"], kw["temporal_window_size"] = a.tas, a.tws
        if not getattr(self, "_cast_done", False):
            # PersonaLive bug on non-default steps: with temporal_adaptive_step < 4 the inner
            # denoising loop runs more than once and DDIM's step returns float32 latents, which
            # the fp16 UNet rejects. Cast the input to the UNet's dtype (no-op on the default path).
            fwd, dt = self.denoising_unet.forward, self.denoising_unet.dtype
            self.denoising_unet.forward = lambda sample, *x, **k: fwd(sample.to(dt), *x, **k)
            # decode latents one frame at a time (diffusers VAE slicing): identical pixels, lower peak
            if a.vae_slicing:
                self.vae.enable_slicing()
            self._cast_done = True
        if a.int8 and not getattr(self, "_int8_done", False):
            quantize_int8(self.denoising_unet)
            self._int8_done = True
        return call(self, *args, **kw)
    pp.Pose2VideoPipeline_Stream.__call__ = lowmem_call
    print(f"[lowmem] temporal_window_size={a.tws} temporal_adaptive_step={a.tas} "
          f"-> {a.tws * a.tas} frames per UNet pass; vae_slicing={a.vae_slicing} int8={a.int8}", flush=True)
    sys.argv = ["inference_offline.py"] + rest
    runpy.run_path(str(PL / "inference_offline.py"), run_name="__main__")
    if a.embed_entry:
        embedder.close()
        write_outputs(out_dir, raw, embedded, embedder.log, a.embed_entry)


def write_outputs(out_dir, raw, embedded, log, sid, fps=25):
    import json
    import subprocess
    import numpy as np
    out_dir.mkdir(parents=True, exist_ok=True)
    for frames, name in ((raw, "personalive__inline_raw.mp4"), (embedded, f"personalive__inline_{sid}.mp4")):
        h, w = frames[0].shape[:2]
        h, w = h - h % 2, w - w % 2          # libx264 / yuv420p need even sizes (the embedder's upscale can give odd ones)
        frames = [f[:h, :w] for f in frames]
        p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                              "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-qp", "0",
                              "-preset", "veryfast", "-pix_fmt", "yuv420p", str(out_dir / name)], stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(np.ascontiguousarray(f).tobytes())
        p.stdin.close()
        p.wait()
    ms = [r["ms"] for r in log]
    (out_dir / f"personalive__inline_{sid}.json").write_text(json.dumps(dict(
        entry=sid, frames=len(log), embed_ms_median=float(np.median(ms)),
        embedded_eyes=sum(r["embedded_eyes"] for r in log), gated_eyes=sum(r["gated_eyes"] for r in log),
        per_frame=log), indent=1))
    print(f"[lowmem] wrote {len(raw)} raw + embedded frames to {out_dir}; embed {np.median(ms):.0f} ms/frame median, "
          f"{sum(r['embedded_eyes'] for r in log)} eyes embedded, {sum(r['gated_eyes'] for r in log)} gated", flush=True)


def quantize_int8(model):
    """Swap nn.Linear layers for bitsandbytes 8-bit layers (weights int8, activations fp16)."""
    import bitsandbytes as bnb
    import torch
    n = 0
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, torch.nn.Linear) and child.in_features >= 256:
                q = bnb.nn.Linear8bitLt(child.in_features, child.out_features, bias=child.bias is not None,
                                        has_fp16_weights=False, threshold=6.0)
                q.weight = bnb.nn.Int8Params(child.weight.data.cpu(), requires_grad=False, has_fp16_weights=False)
                if child.bias is not None:
                    q.bias = torch.nn.Parameter(child.bias.data, requires_grad=False)
                setattr(mod, cname, q.to(child.weight.device))
                n += 1
    torch.cuda.empty_cache()
    print(f"[lowmem] int8: replaced {n} Linear layers in the denoising UNet", flush=True)


if __name__ == "__main__":
    main()
