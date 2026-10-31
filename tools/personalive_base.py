"""Rebuild PersonaLive's missing base model from official sources, and prove it is unused.

PersonaLive loads its UNets from `lambdalabs/sd-image-variations-diffusers`, but that
repository was reset on 2026-10-07 and now holds only a README. What PersonaLive takes
from it, and how this tool replaces each part:

  unet/config.json             architecture only -> the standard SD-1.x UNet2DConditionModel
                               config (cross_attention_dim 768, the SD image-variations UNet)
  unet/diffusion_pytorch_model base weights -> a placeholder (seeded random init). This is
                               only legitimate if PersonaLive's own checkpoints overwrite
                               EVERY weight: checked below for both UNets, with shapes
  image_encoder                CLIP ViT-L/14 vision tower -> `openai/clip-vit-large-patch14`
                               (the image-variations model card names this encoder; it was
                               frozen during their fine-tuning)

Writes the base folder plus `configs/personalive_rebuilt.yaml` (absolute paths), and
exits non-zero if any UNet weight would keep its placeholder value.

    python -s tools/personalive_base.py
"""
import os
import json
import sys
from pathlib import Path

import torch

PL = Path(os.environ.get("PERSONALIVE_DIR", "external/PersonaLive"))
sys.path.insert(0, str(PL))
BASE = PL / "pretrained_weights" / "sd-image-variations-diffusers"
CKPT = PL / "pretrained_weights" / "personalive"
OUT_CFG = Path(__file__).resolve().parent.parent / "configs" / "personalive_rebuilt.yaml"

SD1_UNET = {
    "_class_name": "UNet2DConditionModel", "act_fn": "silu", "attention_head_dim": 8,
    "block_out_channels": [320, 640, 1280, 1280], "center_input_sample": False, "cross_attention_dim": 768,
    "down_block_types": ["CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D", "DownBlock2D"],
    "downsample_padding": 1, "flip_sin_to_cos": True, "freq_shift": 0, "in_channels": 4, "layers_per_block": 2,
    "mid_block_scale_factor": 1, "norm_eps": 1e-05, "norm_num_groups": 32, "out_channels": 4, "sample_size": 64,
    "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"],
}


def coverage(model, *ckpts):
    """Keys of `model` not overwritten (with the right shape) by any checkpoint."""
    want = {k: v.shape for k, v in model.state_dict().items()}
    got = {}
    for c in ckpts:
        got.update({k: v.shape for k, v in torch.load(c, map_location="cpu").items()})
    missing = sorted(k for k in want if k not in got)
    bad_shape = sorted(k for k in want if k in got and tuple(got[k]) != tuple(want[k]))
    return missing, bad_shape


def main():
    from diffusers import UNet2DConditionModel
    from omegaconf import OmegaConf
    from src.models.unet_2d_condition import UNet2DConditionModel as RefUNet   # PersonaLive's: no output head
    from src.models.unet_3d import UNet3DConditionModel

    unet_dir = BASE / "unet"
    unet_dir.mkdir(parents=True, exist_ok=True)
    (unet_dir / "config.json").write_text(json.dumps(SD1_UNET, indent=1))
    torch.manual_seed(0)
    UNet2DConditionModel(**{k: v for k, v in SD1_UNET.items() if not k.startswith("_")}).save_pretrained(
        unet_dir, safe_serialization=True)
    print("wrote", unet_dir)

    # the class inference_offline.py actually uses: conv_norm_out/conv_out are removed,
    # because the reference UNet only writes its attention features to the bank
    ref = RefUNet.from_pretrained(BASE, subfolder="unet")
    m_ref, s_ref = coverage(ref, CKPT / "reference_unet.pth")
    infer = OmegaConf.load(PL / "configs" / "inference" / "inference_stage3.yaml")
    den = UNet3DConditionModel.from_pretrained_2d(BASE, "", subfolder="unet",
                                                  unet_additional_kwargs=infer.unet_additional_kwargs)
    m_den, s_den = coverage(den, CKPT / "denoising_unet.pth", CKPT / "temporal_module.pth")
    print(f"reference_unet: {len(ref.state_dict())} tensors, not covered {len(m_ref)}, shape mismatch {len(s_ref)}")
    print(f"denoising_unet: {len(den.state_dict())} tensors, not covered {len(m_den)}, shape mismatch {len(s_den)}")
    for k in (m_ref + s_ref + m_den + s_den)[:20]:
        print("   ", k)

    clip = PL / "pretrained_weights" / "clip-vit-large-patch14"
    cfg = OmegaConf.load(PL / "configs" / "prompts" / "personalive_offline.yaml")
    cfg.pretrained_base_model_path = str(BASE)
    cfg.image_encoder_path = str(clip)
    cfg.vae_path = str(PL / "pretrained_weights" / "sd-vae-ft-mse")
    cfg.denoising_unet_path = str(CKPT / "denoising_unet.pth")
    cfg.inference_config = str(PL / "configs" / "inference" / "inference_stage3.yaml")
    OUT_CFG.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, OUT_CFG)
    print("wrote", OUT_CFG)
    if m_ref or s_ref or m_den or s_den:
        sys.exit("placeholder weights would remain in use: the rebuilt base is NOT equivalent")
    print("OK: every UNet weight comes from PersonaLive's own checkpoints")


if __name__ == "__main__":
    main()
