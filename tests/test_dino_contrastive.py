"""Test DINOv3 image encoder's ability to distinguish glyph-level differences.

Compares: CLS pooled, patch mean-pool (all), patch mean-pool (bbox only).
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoImageProcessor, AutoModel

POS_IMAGE = "/home/shaush/pairs/1.jpg"
NEG_IMAGE = "/home/shaush/pairs/2.png"
BBOX = [58, 493, 1467, 332]  # COCO [x, y, w, h]
POS_TEXT = "백조명품옷수선"
NEG_TEXT = "백존명품홋수선"
FONT_PATH = "/home/shaush/cortext/NotoSansKR-VariableFont_wght.ttf"
DINO_MODEL = "facebook/dinov3-vitb16-pretrain-lvd1689m"
PATCH_SIZE = 16  # DINOv3 ViT-B/16


def render_text_image(text: str, height: int = 256) -> Image.Image:
    fs = max(height // 3, 48)
    font = ImageFont.truetype(FONT_PATH, fs)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=font)
    tw, th = x1 - x0, y1 - y0
    pad = 64
    w = tw + pad * 2
    h = th + pad * 2
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    sw = max(fs // 30, 1)
    draw.text(((w - tw) // 2 - x0, (h - th) // 2 - y0),
              text, fill="black", font=font, stroke_width=sw, stroke_fill="black")
    return img


def bbox_to_patch_indices(
    bbox: list[int], orig_w: int, orig_h: int,
    proc_w: int, proc_h: int, grid_w: int, grid_h: int,
) -> list[int]:
    sx, sy = proc_w / orig_w, proc_h / orig_h
    x1 = int(bbox[0] * sx) // PATCH_SIZE
    y1 = int(bbox[1] * sy) // PATCH_SIZE
    x2 = math.ceil((bbox[0] + bbox[2]) * sx / PATCH_SIZE)
    y2 = math.ceil((bbox[1] + bbox[3]) * sy / PATCH_SIZE)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(grid_w, x2), min(grid_h, y2)
    indices = []
    for r in range(y1, y2):
        for c in range(x1, x2):
            indices.append(r * grid_w + c)
    return indices


def compute_similarities(
    pos_feat: torch.Tensor, neg_feat: torch.Tensor,
    label: str,
):
    pos_n = F.normalize(pos_feat.float(), dim=-1)
    neg_n = F.normalize(neg_feat.float(), dim=-1)
    sim = (pos_n * neg_n).sum().item()
    dist = (pos_feat.float() - neg_feat.float()).norm().item()
    print(f"  {label:30s} | cos_sim: {sim:.6f} | L2_dist: {dist:.4f}")


def run_pair(
    name: str, img_a: Image.Image, img_b: Image.Image,
    processor, model, bbox_indices: list[int] | None = None,
):
    print(f"\n=== {name} ===")
    inputs_a = processor(images=img_a, return_tensors="pt").to(model.device)
    inputs_b = processor(images=img_b, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out_a = model(**inputs_a)
        out_b = model(**inputs_b)

    # CLS token (pooled)
    compute_similarities(out_a.pooler_output[0], out_b.pooler_output[0], "CLS (pooled)")

    # Patch tokens: last_hidden_state[:, 1:, :] (skip CLS)
    patch_a = out_a.last_hidden_state[0, 1:]
    patch_b = out_b.last_hidden_state[0, 1:]

    compute_similarities(patch_a.mean(dim=0), patch_b.mean(dim=0), "Patch mean-pool (all)")

    if bbox_indices:
        idx = torch.tensor(bbox_indices, device=patch_a.device)
        bbox_a = patch_a[idx].mean(dim=0)
        bbox_b = patch_b[idx].mean(dim=0)
        compute_similarities(bbox_a, bbox_b, f"Patch mean-pool (bbox, {len(bbox_indices)} patches)")


def main():
    cache_dir = "/scratch2/shaush/models"
    print(f"Loading DINOv3: {DINO_MODEL}")
    processor = AutoImageProcessor.from_pretrained(DINO_MODEL, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(DINO_MODEL, cache_dir=cache_dir, device_map="auto")
    model.eval()

    # Determine processor's resize dimensions
    proc_size = processor.size
    proc_h = proc_size.get("height", proc_size.get("shortest_edge", 224))
    proc_w = proc_size.get("width", proc_h)
    grid_h, grid_w = proc_h // PATCH_SIZE, proc_w // PATCH_SIZE
    print(f"Processor size: {proc_w}x{proc_h}, patch grid: {grid_w}x{grid_h}")

    # --- Natural images ---
    pos_img = Image.open(POS_IMAGE).convert("RGB")
    neg_img = Image.open(NEG_IMAGE).convert("RGB")
    orig_w, orig_h = pos_img.size

    bbox_idx = bbox_to_patch_indices(BBOX, orig_w, orig_h, proc_w, proc_h, grid_w, grid_h)
    print(f"Bbox patches: {len(bbox_idx)}/{grid_w * grid_h}")

    run_pair("Natural images (pos vs neg)", pos_img, neg_img,
             processor, model, bbox_indices=bbox_idx)

    # Same image sanity check
    run_pair("Natural images (pos vs pos, sanity)", pos_img, pos_img,
             processor, model, bbox_indices=bbox_idx)

    # --- Rendered text ---
    pos_render = render_text_image(POS_TEXT)
    neg_render = render_text_image(NEG_TEXT)
    if neg_render.size != pos_render.size:
        neg_render = neg_render.resize(pos_render.size, Image.BILINEAR)
    print(f"\nRendered size: {pos_render.size}")

    run_pair("Rendered text (pos vs neg)", pos_render, neg_render, processor, model)
    run_pair("Rendered text (pos vs pos, sanity)", pos_render, pos_render, processor, model)

    print("\nDone.")


if __name__ == "__main__":
    main()
