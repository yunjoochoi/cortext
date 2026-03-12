"""Test OCR recognition hidden states for glyph-level discrimination.

Uses EasyOCR's CRNN recognizer (PyTorch) to extract features before the CTC head
and compare cosine similarity between similar/different Korean text crops.
"""

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

POS_IMAGE = "/home/shaush/pairs/1.jpg"
NEG_IMAGE = "/home/shaush/pairs/2.png"
BBOX = [58, 493, 1467, 332]  # COCO [x, y, w, h]
POS_TEXT = "백조명품옷수선"
NEG_TEXT = "백존명품홋수선"
FONT_PATH = "/home/shaush/cortext/NotoSansKR-VariableFont_wght.ttf"
DEBUG_DIR = Path("/home/shaush/pairs/ppocr_debug")


def crop_bbox(img: Image.Image, bbox: list[int]) -> Image.Image:
    x, y, w, h = bbox
    return img.crop((x, y, x + w, y + h))


def render_text(text: str, height: int = 64) -> Image.Image:
    fs = max(height - 16, 32)
    font = ImageFont.truetype(FONT_PATH, fs)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=font)
    tw, th = x1 - x0, y1 - y0
    pad = 16
    img = Image.new("RGB", (tw + pad * 2, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((pad - x0, (height - th) // 2 - y0), text, fill="black", font=font)
    return img


def preprocess_for_crnn(img: Image.Image, target_h: int = 64) -> np.ndarray:
    w, h = img.size
    ratio = target_h / h
    target_w = max(int(w * ratio), 1)
    img = img.resize((target_w, target_h), Image.BILINEAR).convert("L")
    arr = np.array(img).astype("float32") / 255.0
    arr = (arr - 0.5) / 0.5
    arr = arr[np.newaxis]  # (1, H, W) grayscale
    return arr


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a = F.normalize(a.float().flatten(), dim=0)
    b = F.normalize(b.float().flatten(), dim=0)
    return (a * b).sum().item()


def l2_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).norm().item()


def extract_crnn_features(recognizer, img_arr: np.ndarray, device: torch.device) -> dict:
    """Extract intermediate features from EasyOCR's CRNN recognizer."""
    model = recognizer.module
    model.eval()

    tensor = torch.from_numpy(img_arr).unsqueeze(0).to(device)  # (1, 3, H, W)

    feats = {}

    # Hook into different stages
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                feats[name] = output.detach()
            elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                feats[name] = output[0].detach()
        return hook_fn

    # Register hooks on key layers
    # CRNN structure: FeatureExtraction (CNN) -> SequenceModeling (BiLSTM) -> Prediction (Linear)
    if hasattr(model, "FeatureExtraction"):
        hooks.append(model.FeatureExtraction.register_forward_hook(make_hook("backbone")))
    if hasattr(model, "SequenceModeling"):
        hooks.append(model.SequenceModeling.register_forward_hook(make_hook("sequence")))
    if hasattr(model, "Prediction"):
        hooks.append(model.Prediction.register_forward_hook(make_hook("prediction")))
    # Also try AdaptiveAvgPool if present
    if hasattr(model, "AdaptiveAvgPool"):
        hooks.append(model.AdaptiveAvgPool.register_forward_hook(make_hook("pool")))

    with torch.no_grad():
        out = model(tensor, text="")

    for h in hooks:
        h.remove()

    return feats


def compare_pair(feats_a: dict, feats_b: dict, label: str):
    print(f"\n--- {label} ---")
    print(f"  {'layer':25s} | {'shape':20s} | {'cos_sim':>10s} | {'L2_dist':>10s}")
    print(f"  {'-'*25}-+-{'-'*20}-+-{'-'*10}-+-{'-'*10}")

    for key in ["backbone", "pool", "sequence", "prediction"]:
        if key not in feats_a or key not in feats_b:
            continue
        a, b = feats_a[key], feats_b[key]

        # Mean-pool to single vector
        if a.dim() == 4:  # (B, C, H, W) -> mean over spatial
            a_pool = a[0].mean(dim=(1, 2))
            b_pool = b[0].mean(dim=(1, 2))
        elif a.dim() == 3:  # (B, T, D) -> mean over time
            a_pool = a[0].mean(dim=0)
            b_pool = b[0].mean(dim=0)
        elif a.dim() == 2:  # (B, D)
            a_pool = a[0]
            b_pool = b[0]
        else:
            continue

        sim = cosine_sim(a_pool, b_pool)
        dist = l2_dist(a_pool, b_pool)
        print(f"  {key:25s} | {str(list(a.shape)):20s} | {sim:10.6f} | {dist:10.4f}")

        # For sequence features, also compare per-timestep
        if key == "sequence" and a.dim() == 3:
            T = min(a.shape[1], b.shape[1])
            per_t_sims = []
            for t in range(T):
                s = cosine_sim(a[0, t], b[0, t])
                per_t_sims.append(s)
            avg_t = np.mean(per_t_sims)
            min_t = np.min(per_t_sims)
            print(f"  {'  seq per-timestep':25s} | {'avg/min':20s} | {avg_t:10.6f} | {min_t:10.6f}")


def main():
    import easyocr

    print("=== OCR Hidden State Contrastive Test (EasyOCR CRNN) ===\n")

    # Load EasyOCR with Korean support
    reader = easyocr.Reader(["ko"], gpu=True, verbose=False)
    recognizer = reader.recognizer
    device = next(recognizer.module.parameters()).device
    print(f"Device: {device}")
    print(f"Model type: {type(recognizer.module)}")

    # Print model structure summary
    print("\nModel structure:")
    for name, child in recognizer.module.named_children():
        params = sum(p.numel() for p in child.parameters())
        print(f"  {name}: {type(child).__name__} ({params:,} params)")

    # Load images
    pos_img = Image.open(POS_IMAGE).convert("RGB")
    neg_img = Image.open(NEG_IMAGE).convert("RGB")
    pos_crop = crop_bbox(pos_img, BBOX)
    neg_crop = crop_bbox(neg_img, BBOX)
    pos_render = render_text(POS_TEXT)
    neg_render = render_text(NEG_TEXT)

    print(f"\nPos crop: {pos_crop.size}, Neg crop: {neg_crop.size}")
    print(f"Pos render: {pos_render.size}, Neg render: {neg_render.size}")

    # Save debug
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    pos_crop.save(DEBUG_DIR / "pos_crop.jpg")
    neg_crop.save(DEBUG_DIR / "neg_crop.jpg")
    pos_render.save(DEBUG_DIR / "pos_render.jpg")
    neg_render.save(DEBUG_DIR / "neg_render.jpg")

    # Preprocess
    pos_crop_arr = preprocess_for_crnn(pos_crop)
    neg_crop_arr = preprocess_for_crnn(neg_crop)
    pos_render_arr = preprocess_for_crnn(pos_render)
    neg_render_arr = preprocess_for_crnn(neg_render)

    # Also run OCR to see what it reads
    print("\n=== OCR Recognition Results ===")
    for name, img in [("pos_crop", pos_crop), ("neg_crop", neg_crop),
                       ("pos_render", pos_render), ("neg_render", neg_render)]:
        result = reader.readtext(np.array(img), detail=0)
        print(f"  {name:15s}: {result}")

    # Extract and compare features
    print("\n=== Feature Comparison ===")

    # Natural image crops
    pos_feats = extract_crnn_features(recognizer, pos_crop_arr, device)
    neg_feats = extract_crnn_features(recognizer, neg_crop_arr, device)
    san_feats = extract_crnn_features(recognizer, pos_crop_arr, device)

    compare_pair(pos_feats, neg_feats, "Natural crop: pos vs neg")
    compare_pair(pos_feats, san_feats, "Natural crop: pos vs pos (sanity)")

    # Rendered text
    pos_r_feats = extract_crnn_features(recognizer, pos_render_arr, device)
    neg_r_feats = extract_crnn_features(recognizer, neg_render_arr, device)
    san_r_feats = extract_crnn_features(recognizer, pos_render_arr, device)

    compare_pair(pos_r_feats, neg_r_feats, "Rendered: pos vs neg")
    compare_pair(pos_r_feats, san_r_feats, "Rendered: pos vs pos (sanity)")

    print("\nDone.")


if __name__ == "__main__":
    main()
