"""Band-only diffusion refinement of an embedded eye (D36 experiment, research/04 section 2.2).

A pretrained inpainting model (SD 1.5 inpaint) redraws ONLY a band that carries no donor texture: the limbal
ring just outside the iris and a strip along the eyelid margin. The iris interior is never in the mask and is
pasted back in pixel space afterwards, so identity still travels only as pixels (P2). The crop is upscaled so the
iris spans ~390 px at the VAE (D21: identity survives a VAE round trip from ~120 px of iris).

    refiner = BandRefiner()                      # loads the weights once (fp16, ~3 GB VRAM)
    frame2 = refiner(frame_embedded, seg)        # seg: the eye's IrisSegmentation from the raw frame
"""
from __future__ import annotations

import os

import cv2
import numpy as np

MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
PROMPT = ("a close-up photograph of a real human eye, natural dark limbal ring, soft eyelid shadow on the iris, "
          "realistic cornea, sharp focus")
NEG = "cartoon, painting, drawing, deformed, blurry, extra pupil, text, watermark"
SIDE = 512
IRIS_FRAC = 0.88        # nothing inside this fraction of the limbus radius is ever redrawn


def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


class BandRefiner:
    def __init__(self, strength=None, steps=None, guidance=None, seed=0, ring=(0.88, 1.10), lid_band=0.10):
        import torch
        from diffusers import StableDiffusionInpaintPipeline
        self.torch = torch
        self.pipe = StableDiffusionInpaintPipeline.from_pretrained(
            MODEL, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, safety_checker=None, requires_safety_checker=False).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        self.strength = float(os.environ.get("IRIS_BAND_STRENGTH", 0.7) if strength is None else strength)
        self.steps = int(os.environ.get("IRIS_BAND_STEPS", 25) if steps is None else steps)
        self.guidance = float(os.environ.get("IRIS_BAND_GUIDANCE", 5.0) if guidance is None else guidance)
        self.seed, self.ring, self.lid_band = seed, ring, lid_band

    def masks(self, seg, S):
        """Band mask (float 0..1) in the SIDE x SIDE crop, built from the segmentation geometry."""
        k = SIDE / S
        R = float(np.mean(seg.iris[2:3])) * k
        yy, xx = np.mgrid[0:SIDE, 0:SIDE]
        d = np.hypot(xx - SIDE / 2, yy - SIDE / 2)
        ring = _smooth(self.ring[0] * R - 2, self.ring[0] * R + 2, d) * (1 - _smooth(self.ring[1] * R - 2, self.ring[1] * R + 2, d))
        return ring, R, d

    def __call__(self, frame, seg, lid_polygon=None):
        from PIL import Image
        cx, cy = float(seg.iris[0]), float(seg.iris[1])
        R0 = float(np.mean(seg.iris[2:3]))
        S = int(round(2.6 * R0)) // 2 * 2
        x0, y0 = int(round(cx - S / 2)), int(round(cy - S / 2))
        H, W = frame.shape[:2]
        pl, pt, pr_, pb = max(0, -x0), max(0, -y0), max(0, x0 + S - W), max(0, y0 + S - H)
        big = cv2.copyMakeBorder(frame, pt, pb, pl, pr_, cv2.BORDER_REPLICATE) if (pl or pt or pr_ or pb) else frame
        ox, oy = x0 + pl, y0 + pt
        crop = big[oy:oy + S, ox:ox + S]
        up = cv2.resize(crop, (SIDE, SIDE), interpolation=cv2.INTER_CUBIC)

        band, R, d = self.masks(seg, S)
        k = SIDE / S
        # eyelid margin: a strip along the lid polygon boundary (the upper-lid shadow lives there)
        poly = seg.lid_polygon if lid_polygon is None else lid_polygon
        lid = np.zeros((SIDE, SIDE), np.uint8)
        pts = np.round((np.asarray(poly, np.float64) - [x0, y0]) * k).astype(np.int32)
        cv2.fillPoly(lid, [pts], 1)
        w = max(2, int(self.lid_band * R))
        edge = (cv2.dilate(lid, np.ones((2 * w + 1, 2 * w + 1), np.uint8)) - cv2.erode(lid, np.ones((2 * w + 1, 2 * w + 1), np.uint8))).astype(np.float32)
        lid_band = edge * (d < 1.25 * R) * (d > self.ring[0] * R)         # never inside the iris
        mask = np.clip(band + lid_band, 0, 1)
        mask = cv2.GaussianBlur(mask, (0, 0), 2.0) * (d > (IRIS_FRAC - 0.02) * R)   # hard guarantee: interior untouched

        gen = self.torch.Generator("cuda").manual_seed(self.seed)
        out = self.pipe(prompt=PROMPT, negative_prompt=NEG, image=Image.fromarray(up),
                        mask_image=Image.fromarray(np.uint8(255 * (mask > 0.05))), height=SIDE, width=SIDE,
                        strength=self.strength, num_inference_steps=self.steps, guidance_scale=self.guidance,
                        generator=gen).images[0]
        out = np.asarray(out)
        if _is_gray(frame):
            g = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
            out = np.repeat(g[..., None], 3, -1)
        sd = cv2.resize(out, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32)
        a = cv2.resize(mask, (S, S), interpolation=cv2.INTER_AREA)[..., None]
        merged = a * sd + (1 - a) * crop.astype(np.float32)
        res = big.copy()
        res[oy:oy + S, ox:ox + S] = np.clip(merged + 0.5, 0, 255).astype(np.uint8)
        return res[pt:pt + H, pl:pl + W]


def _is_gray(rgb):
    s = rgb[:: max(1, rgb.shape[0] // 64), :: max(1, rgb.shape[1] // 64)].astype(np.int16)
    return max(np.abs(s[..., 0] - s[..., 1]).mean(), np.abs(s[..., 1] - s[..., 2]).mean()) < 6.0
