"""Test contrastive discriminability: mean-pool vs bbox-crop vs rendered text."""

import copy
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from diffusers.training_utils import compute_density_for_timestep_sampling
from transformers import Qwen2Tokenizer, Qwen3Model

MODEL_PATH = "/scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
POS_IMAGE = "/home/shaush/pairs/1.jpg"
NEG_IMAGE = "/home/shaush/pairs/2.png"
PROMPT = "A signage photo, texts are written on it: '백조명품옷수선'"
BBOX = [58, 493, 1467, 332]  # COCO format: [x, y, w, h]
POS_TEXT = "백조명품옷수선"
NEG_TEXT = "백존명품홋수선"
FONT_PATH = "/home/shaush/cortext/NotoSansKR-VariableFont_wght.ttf"

DEVICE = "cuda"
DTYPE = torch.bfloat16
TEMPERATURE = 0.07
LAYERS_TO_PROBE = list(range(0, 30, 2))
VAE_SCALE_FACTOR = 8
PATCH_SIZE = 2
LATENT_STRIDE = VAE_SCALE_FACTOR * PATCH_SIZE  # 16
ALIGN = 16


class LayerCapture:
    def __init__(self):
        self.stored: dict[str, torch.Tensor] = {}
        self._hooks = []

    def register(self, transformer, layer_indices):
        for idx in layer_indices:
            layer = transformer.layers[idx]

            def _make_hook(layer_idx):
                def hook(module, args):
                    self.stored[f"x_{layer_idx}"] = args[0]
                return hook

            self._hooks.append(layer.register_forward_pre_hook(_make_hook(idx)))

    def clear(self):
        self.stored.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def fit_size(w: int, h: int, max_pixels: int = 1024 * 1024) -> tuple[int, int]:
    if w * h > max_pixels:
        scale = (max_pixels / (w * h)) ** 0.5
        w, h = int(w * scale), int(h * scale)
    w = w // ALIGN * ALIGN
    h = h // ALIGN * ALIGN
    return max(ALIGN, w), max(ALIGN, h)


def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    new_w, new_h = fit_size(*img.size)
    if (new_w, new_h) != img.size:
        img = img.resize((new_w, new_h), Image.BILINEAR)
    return img


def bbox_to_patch_indices(
    bbox: list[int], orig_w: int, orig_h: int,
    resized_w: int, resized_h: int, grid_w: int, grid_h: int,
) -> list[int]:
    """COCO bbox [x,y,w,h] in original pixel space → flat patch token indices."""
    sx, sy = resized_w / orig_w, resized_h / orig_h
    x1 = int(bbox[0] * sx) // LATENT_STRIDE
    y1 = int(bbox[1] * sy) // LATENT_STRIDE
    x2 = math.ceil((bbox[0] + bbox[2]) * sx / LATENT_STRIDE)
    y2 = math.ceil((bbox[1] + bbox[3]) * sy / LATENT_STRIDE)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(grid_w, x2), min(grid_h, y2)
    indices = []
    for r in range(y1, y2):
        for c in range(x1, x2):
            indices.append(r * grid_w + c)
    return indices


def render_text_image(text: str, base_height: int = 256) -> Image.Image:
    """Render black text on white, matching contrastive training setup."""
    fs = max(base_height // 3, 48)
    font = ImageFont.truetype(FONT_PATH, fs)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=font)
    tw, th = x1 - x0, y1 - y0
    padding = 64
    w = (tw + padding * 2 + ALIGN - 1) // ALIGN * ALIGN
    h = (th + padding * 2 + ALIGN - 1) // ALIGN * ALIGN
    w, h = max(ALIGN, w), max(ALIGN, h)
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    x = (w - tw) // 2 - x0
    y = (h - th) // 2 - y0
    sw = max(fs // 30, 1)
    draw.text((x, y), text, fill="black", font=font, stroke_width=sw, stroke_fill="black")
    return img


def compute_sim(
    x: torch.Tensor, img_indices: list[int] | None,
    img_seq_len: int, txt_start: int,
) -> tuple[float, float, float, float]:
    """Compute pos/neg similarity and loss for one layer."""
    if img_indices is not None:
        idx = torch.tensor(img_indices, device=x.device)
        pos_img_h = x[0, idx].mean(dim=0)
        neg_img_h = x[1, idx].mean(dim=0)
    else:
        pos_img_h = x[0, :img_seq_len].mean(dim=0)
        neg_img_h = x[1, :img_seq_len].mean(dim=0)
    txt_h = x[0, txt_start:].mean(dim=0)

    pos_img_h = F.normalize(pos_img_h.float(), dim=-1)
    neg_img_h = F.normalize(neg_img_h.float(), dim=-1)
    txt_h = F.normalize(txt_h.float(), dim=-1)

    pos_sim = (pos_img_h * txt_h).sum()
    neg_sim = (neg_img_h * txt_h).sum()
    logits = torch.stack([pos_sim, neg_sim]).unsqueeze(0) / TEMPERATURE
    labels = torch.zeros(1, dtype=torch.long, device=x.device)
    loss = F.cross_entropy(logits, labels).item()
    return pos_sim.item(), neg_sim.item(), (pos_sim - neg_sim).item(), loss


def run_experiment(
    name: str, pos_img: Image.Image, neg_img: Image.Image, prompt: str,
    pipeline, vae, transformer, capture, to_tensor,
    vae_shift: float, vae_scale: float,
    bbox_indices: list[int] | None = None,
):
    pw, ph = pos_img.size
    img_seq_len = (ph // LATENT_STRIDE) * (pw // LATENT_STRIDE)

    pv = torch.stack([to_tensor(pos_img), to_tensor(neg_img)]).to(DEVICE, dtype=DTYPE)
    with torch.no_grad():
        latents = vae.encode(pv).latent_dist.mode()
    latents = (latents - vae_shift) * vae_scale

    scheduler_copy = copy.deepcopy(pipeline.scheduler)
    u = compute_density_for_timestep_sampling("none", batch_size=1)
    idx = (u * scheduler_copy.config.num_train_timesteps).long()
    timestep = scheduler_copy.timesteps[idx].to(DEVICE)
    sigmas = scheduler_copy.sigmas.to(DEVICE, dtype=DTYPE)
    schedule_ts = scheduler_copy.timesteps.to(DEVICE)
    step_idx = (schedule_ts == timestep).nonzero().item()
    sigma = sigmas[step_idx].reshape(1, 1, 1, 1)

    noise = torch.randn_like(latents)
    noisy = (1.0 - sigma) * latents + sigma * noise
    ts_norm = (1000 - timestep) / 1000

    with torch.no_grad():
        prompt_embeds, _ = pipeline.encode_prompt(
            prompt=[prompt], do_classifier_free_guidance=False, max_sequence_length=512)
    prompt_list = [prompt_embeds[0], prompt_embeds[0]]

    capture.clear()
    noisy_list = list(noisy.unsqueeze(2).unbind(dim=0))
    ts_expanded = ts_norm.expand(2)
    with torch.no_grad():
        _ = transformer(noisy_list, ts_expanded, prompt_list, return_dict=False)

    modes = [("mean-pool(all)", None)]
    if bbox_indices:
        modes.append(("bbox-only", bbox_indices))

    for mode_name, indices in modes:
        print(f"\n=== {name} / {mode_name} ===")
        print(f" Layer |  pos_sim |  neg_sim |      gap |     loss")
        print("-" * 56)
        for layer_idx in LAYERS_TO_PROBE:
            x = capture.stored.get(f"x_{layer_idx}")
            if x is None or x.shape[0] < 2:
                continue
            ps, ns, gap, loss = compute_sim(x, indices, img_seq_len, img_seq_len)
            print(f"L{layer_idx:>4}  | {ps:>8.4f} | {ns:>8.4f} | {gap:>+8.4f} | {loss:>8.4f}")
        print(f"  ln(2) = 0.6931 (random baseline)")


def main():
    print("Loading models...")
    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(MODEL_PATH, subfolder="vae")
    transformer = ZImageTransformer2DModel.from_pretrained(MODEL_PATH, subfolder="transformer", torch_dtype=DTYPE)
    text_encoder = Qwen3Model.from_pretrained(MODEL_PATH, subfolder="text_encoder")

    vae.requires_grad_(False).to(DEVICE, dtype=DTYPE)
    transformer.requires_grad_(False).to(DEVICE, dtype=DTYPE)
    text_encoder.requires_grad_(False).to(DEVICE, dtype=DTYPE)

    vae_shift = vae.config.shift_factor
    vae_scale = vae.config.scaling_factor

    pipeline = ZImagePipeline(
        vae=vae, transformer=transformer, tokenizer=tokenizer,
        text_encoder=text_encoder, scheduler=scheduler,
    )
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    capture = LayerCapture()
    capture.register(transformer, LAYERS_TO_PROBE)

    # --- Experiment 1: Natural images (mean-pool all + bbox-only) ---
    pos_img = load_image(POS_IMAGE)
    neg_img = load_image(NEG_IMAGE)
    orig_img = Image.open(POS_IMAGE).convert("RGB")
    orig_w, orig_h = orig_img.size
    pw, ph = pos_img.size
    if neg_img.size != pos_img.size:
        neg_img = neg_img.resize(pos_img.size, Image.BILINEAR)

    grid_w, grid_h = pw // LATENT_STRIDE, ph // LATENT_STRIDE
    bbox_idx = bbox_to_patch_indices(BBOX, orig_w, orig_h, pw, ph, grid_w, grid_h)
    print(f"Image: {pw}x{ph}, grid: {grid_w}x{grid_h}, bbox patches: {len(bbox_idx)}/{grid_w * grid_h}")

    # Save bbox visualization
    debug_dir = Path("/home/shaush/pairs/debug")
    debug_dir.mkdir(exist_ok=True)
    sx, sy = pw / orig_w, ph / orig_h
    rx1 = int(BBOX[0] * sx)
    ry1 = int(BBOX[1] * sy)
    rx2 = int((BBOX[0] + BBOX[2]) * sx)
    ry2 = int((BBOX[1] + BBOX[3]) * sy)
    for tag, img in [("pos", pos_img), ("neg", neg_img)]:
        vis = img.copy()
        draw = ImageDraw.Draw(vis)
        draw.rectangle([rx1, ry1, rx2, ry2], outline="red", width=3)
        for idx in bbox_idx:
            r, c = divmod(idx, grid_w)
            px1 = c * LATENT_STRIDE
            py1 = r * LATENT_STRIDE
            draw.rectangle([px1, py1, px1 + LATENT_STRIDE, py1 + LATENT_STRIDE],
                           outline="blue", width=1)
        vis.save(debug_dir / f"bbox_patches_{tag}.jpg")
    print(f"Saved bbox visualizations: {debug_dir}/bbox_patches_pos.jpg, bbox_patches_neg.jpg")

    run_experiment(
        "Natural image", pos_img, neg_img, PROMPT,
        pipeline, vae, transformer, capture, to_tensor,
        vae_shift, vae_scale, bbox_indices=bbox_idx,
    )

    # --- Experiment 2: Rendered text (matching contrastive training) ---
    pos_render = render_text_image(POS_TEXT)
    neg_render = render_text_image(NEG_TEXT)
    if neg_render.size != pos_render.size:
        neg_render = neg_render.resize(pos_render.size, Image.BILINEAR)
    rw, rh = pos_render.size
    print(f"\nRendered size: {rw}x{rh}")

    # Save rendered images
    pos_render.save(debug_dir / "render_pos.jpg")
    neg_render.save(debug_dir / "render_neg.jpg")
    print(f"Saved rendered images: {debug_dir}/render_pos.jpg, render_neg.jpg")

    render_prompt = f"The text '{POS_TEXT}'"
    run_experiment(
        "Rendered text", pos_render, neg_render, render_prompt,
        pipeline, vae, transformer, capture, to_tensor,
        vae_shift, vae_scale,
    )

    capture.remove()
    print("\nDone.")


if __name__ == "__main__":
    main()
