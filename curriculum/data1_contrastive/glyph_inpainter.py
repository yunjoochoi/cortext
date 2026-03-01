"""Glyph rendering + Paint-by-Example inpainting for hard negative generation.

Model: Fantasy-Studio/Paint-by-Example (CVPR 2023)
  - Reference image guided inpainting
  - Takes: source image + binary mask + reference glyph → inpainted output
  - VRAM: ~8GB FP16

Workflow:
  1. render_glyph(target_syllable)   → clean white-bg reference image
  2. estimate_syllable_bbox(...)     → pixel bbox of syllable within crop
  3. inpaint_syllable(crop, ...)     → edited crop with syllable replaced
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _find_korean_font() -> str | None:
    """Search system font directories for a Korean font."""
    import os
    search_dirs = [
        "/usr/share/fonts", "/usr/local/share/fonts",
        Path.home() / ".fonts", Path.home() / ".local/share/fonts",
    ]
    keywords = ("nanum", "malgun", "batang", "gulim", "dotum", "nanumgothic")
    for d in search_dirs:
        for root, _, files in os.walk(d):
            for fname in files:
                if fname.lower().endswith((".ttf", ".otf")):
                    if any(k in fname.lower() for k in keywords):
                        return str(Path(root) / fname)
    return None


_KOREAN_FONT_PATH: str | None = _find_korean_font()


def render_glyph(text: str, height: int = 128) -> np.ndarray:
    """Render text on white background with a Korean font.

    Returns RGB uint8 array of shape (height, width, 3).
    Width is determined by text length to maintain natural aspect ratio.
    """
    font_size = int(height * 0.75)
    try:
        font = (ImageFont.truetype(_KOREAN_FONT_PATH, font_size)
                if _KOREAN_FONT_PATH else ImageFont.load_default())
    except Exception:
        font = ImageFont.load_default()

    # Measure text width
    dummy = Image.new("RGB", (1, 1))
    draw  = ImageDraw.Draw(dummy)
    bbox  = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = int(height * 0.1)
    img_w = max(text_w + pad * 2, height)
    img_h = height

    img  = Image.new("RGB", (img_w, img_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    x = (img_w - text_w) // 2 - bbox[0]
    y = (img_h - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    return np.array(img, dtype=np.uint8)


def estimate_syllable_bbox(crop_w: int, crop_h: int,
                           text: str, syl_idx: int,
                           padding_ratio: float = 0.05) -> tuple[int, int, int, int]:
    """Estimate pixel bbox (x, y, w, h) of a syllable using equal-width assumption.

    Adds a small horizontal padding to account for inaccuracy.
    """
    n = max(len(text), 1)
    syl_w = crop_w / n
    pad   = int(crop_w * padding_ratio)
    x = max(0, int(syl_idx * syl_w) - pad)
    w = min(int(syl_w) + 2 * pad, crop_w - x)
    return x, 0, w, crop_h


def _build_mask(crop_h: int, crop_w: int,
                bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Binary mask: 255 in the region to inpaint, 0 elsewhere."""
    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    x, y, w, h = bbox
    mask[y:y+h, x:x+w] = 255
    return mask


def load_pbe_model(device: str = "cuda"):
    """Load Paint-by-Example pipeline (call once, reuse)."""
    from diffusers import PaintByExamplePipeline
    import torch
    dtype = torch.float16 if "cuda" in device else torch.float32
    pipe = PaintByExamplePipeline.from_pretrained(
        "Fantasy-Studio/Paint-by-Example",
        torch_dtype=dtype,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def inpaint_syllable(
    crop_bgr: np.ndarray,
    syl_idx: int,
    text: str,
    target_syl: str,
    model,
    num_inference_steps: int = 30,
) -> np.ndarray:
    """Replace one syllable in crop_bgr with target_syl using Paint-by-Example.

    Args:
        crop_bgr:    source image crop (BGR uint8)
        syl_idx:     index of syllable to replace within text
        text:        original text of the crop (e.g. "편의점")
        target_syl:  single target syllable (e.g. "펀")
        model:       PaintByExamplePipeline
        num_inference_steps: diffusion steps

    Returns:
        Edited crop as BGR uint8 array (same shape as input).
    """
    h, w = crop_bgr.shape[:2]
    bbox = estimate_syllable_bbox(w, h, text, syl_idx)
    mask_arr = _build_mask(h, w, bbox)

    source_rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    source_pil  = Image.fromarray(source_rgb)
    mask_pil    = Image.fromarray(mask_arr)

    ref_arr = render_glyph(target_syl, height=max(h, 64))
    ref_pil = Image.fromarray(ref_arr)

    result = model(
        image=source_pil,
        mask_image=mask_pil,
        example_image=ref_pil,
        num_inference_steps=num_inference_steps,
    ).images[0]

    result_bgr = cv2.cvtColor(np.array(result, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(result_bgr, (w, h))
