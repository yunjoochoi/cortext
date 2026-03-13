"""Z-Image LoRA fine-tuning on manifest.jsonl
L_mse(전체) + λ₁ * L_char(글자 위치) + λ₂ * L_contrastive(glyph discriminability)

accelerate launch training/train_lora_z_image_simple.py \
    --pretrained_model_name_or_path /scratch2/shaush/models/models--Tongyi-MAI--Z-Image \
    --manifest /scratch2/shaush/coreset_output/manifest.jsonl \
    --output_dir /scratch2/shaush/training_output/lora_simple \
    --max_pixels 1048576 \
    --train_batch_size 1 --gradient_accumulation_steps 4 \
    --max_train_steps 5000 --learning_rate 1e-4 \
    --rank 32 --mixed_precision bf16 \
    --gradient_checkpointing \
    --checkpointing_steps 500
"""

import argparse
import copy
import json
import logging
import math
import os
import random
import shutil
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL import ImageDraw, ImageFont
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from core.jamo import substitute_per_char

from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _collate_lora_metadata,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from transformers import Qwen2Tokenizer, Qwen3Model

# Hard-negative generation for contrastive loss
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="z-image-lora-simple")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pixels", type=int, default=1024*1024,
                   help="Max total pixels; images are scaled down to fit (aspect ratio preserved, 16-aligned)")
    p.add_argument("--train_batch_size", type=int, default=1)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--checkpoints_total_limit", type=int, default=10)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--offload", action="store_true")
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--weighting_scheme", type=str, default="none",
                   choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    p.add_argument("--logit_mean", type=float, default=0.0)
    p.add_argument("--logit_std", type=float, default=1.0)
    p.add_argument("--mode_scale", type=float, default=1.29)
    p.add_argument("--validation_prompt", type=str, default=None)
    p.add_argument("--validation_epochs", type=int, default=50)
    p.add_argument("--num_validation_images", type=int, default=4)
    p.add_argument("--cache_latents", action="store_true")
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--char_loss_lambda", type=float, default=0.05,
                   help="Weight for character localization loss (0 to disable)")
    p.add_argument("--char_loss_layers", type=str, default="12,13,14,15,16",
                   help="Comma-separated transformer layer indices for char localization loss")
    # Contrastive glyph discriminability loss
    p.add_argument("--contrastive_lambda", type=float, default=0.05,
                   help="Weight for contrastive glyph discriminability loss (0 to disable)")
    p.add_argument("--contrastive_layers", type=str, default="14,15,16",
                   help="Comma-separated transformer layer indices for contrastive loss")
    p.add_argument("--contrastive_temperature", type=float, default=0.07,
                   help="Temperature for InfoNCE contrastive loss")
    p.add_argument("--font_path", type=str, default=None,
                   help="Path to Korean TTF font for synthetic text rendering")
    p.add_argument("--synthetic_size", type=int, default=256,
                   help="Resolution for synthetic rendered text images (must be divisible by 16)")
    return p.parse_args()


def build_prompt(caption: str, texts: list) -> str:
    text_str = ", ".join(texts)
    if caption:
        return f"{caption}, texts are written on it: {text_str}"
    return f"A signage photo, texts are written on it: {text_str}"


# ---------------------------------------------------------------------------
# Dataset: reads manifest.jsonl, builds prompt from caption + text
# ---------------------------------------------------------------------------
class ManifestDataset(Dataset):
    ALIGN = 16  # must be divisible by vae_scale_factor(8) * patch_size(2)

    def __init__(self, manifest_path: str, max_pixels: int = 1024 * 1024):
        self.records = []
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("text"):
                    self.records.append(rec)
        self.max_pixels = max_pixels
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def _fit_size(self, w: int, h: int) -> tuple[int, int]:
        """Scale down to fit max_pixels, then floor-align to ALIGN."""
        if w * h > self.max_pixels:
            scale = (self.max_pixels / (w * h)) ** 0.5
            w, h = int(w * scale), int(h * scale)
        w = w // self.ALIGN * self.ALIGN
        h = h // self.ALIGN * self.ALIGN
        return max(self.ALIGN, w), max(self.ALIGN, h)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image_path"])
        image = exif_transpose(image)
        orig_w, orig_h = image.size  # PIL: (w, h)
        if image.mode != "RGB":
            image = image.convert("RGB")

        new_w, new_h = self._fit_size(orig_w, orig_h)
        if (new_w, new_h) != (orig_w, orig_h):
            image = image.resize((new_w, new_h), Image.BILINEAR)

        texts = rec["text"] if isinstance(rec["text"], list) else [rec["text"]]
        bboxes = rec.get("bbox", {})
        return {
            "pixel_values": self.to_tensor(image),
            "prompt": build_prompt(rec.get("caption", ""), texts),
            "texts": texts,
            "bboxes": bboxes,
            "orig_size": (orig_w, orig_h),
            "target_size": (new_w, new_h),
        }


def collate_fn(examples):
    # batch_size=1 assumed; variable resolution per sample
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {
        "pixel_values": pixel_values,
        "prompts": [e["prompt"] for e in examples],
        "texts_list": [e["texts"] for e in examples],
        "bboxes_list": [e["bboxes"] for e in examples],
        "orig_sizes": [e["orig_size"] for e in examples],
        "target_sizes": [e["target_size"] for e in examples],
    }


# ---------------------------------------------------------------------------
# Character localization loss helpers
# ---------------------------------------------------------------------------
def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """RoPE identical to ZSingleStreamAttnProcessor."""
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        fc = freqs_cis.unsqueeze(2)  # don't mutate the shared tensor
        x_out = torch.view_as_real(x * fc).flatten(3)
        return x_out.type_as(x_in)


def create_spatial_masks(bboxes_list, orig_sizes, target_h, target_w, patch_h, patch_w, device):
    """Convert bboxes [x,y,w,h] to binary masks at patch-grid resolution."""
    batch_masks = []
    for bboxes, (orig_w, orig_h) in zip(bboxes_list, orig_sizes):
        masks = {}
        sx, sy = target_w / orig_w, target_h / orig_h
        dx, dy = target_w / patch_w, target_h / patch_h  # pixels per patch cell
        for text, (bx, by, bw, bh) in bboxes.items():
            x1 = max(0, int(bx * sx / dx))
            y1 = max(0, int(by * sy / dy))
            x2 = min(patch_w, int((bx + bw) * sx / dx) + 1)
            y2 = min(patch_h, int((by + bh) * sy / dy) + 1)
            mask = torch.zeros(patch_h, patch_w, device=device)
            if y2 > y1 and x2 > x1:
                mask[y1:y2, x1:x2] = 1.0
            masks[text] = mask
        batch_masks.append(masks)
    return batch_masks


def find_text_token_indices(tokenizer, prompt, texts, max_length):
    """Find which token positions in the tokenized prompt correspond to each text.

    Uses subsequence matching on token IDs.  Falls back to single-char tokenization
    when the context-dependent BPE split differs from isolated tokenization.
    """
    full_ids = tokenizer.encode(prompt, max_length=max_length, truncation=True)
    result = {}
    used_positions: set[int] = set()
    # Sort by length descending to prevent substring overlap
    # e.g. "보광약국" matched before "보광"
    for text in sorted(texts, key=len, reverse=True):
        text_ids = tokenizer.encode(text, add_special_tokens=False)
        if not text_ids:
            continue
        n = len(text_ids)
        found = False
        for i in range(len(full_ids) - n + 1):
            positions = set(range(i, i + n))
            if positions & used_positions:
                continue
            if full_ids[i:i + n] == text_ids:
                result[text] = list(range(i, i + n))
                used_positions.update(positions)
                found = True
                break
        if not found:
            # Fallback: try matching individual characters
            char_indices = []
            search_start = 0
            for ch in text:
                ch_ids = tokenizer.encode(ch, add_special_tokens=False)
                if not ch_ids:
                    continue
                m = len(ch_ids)
                for j in range(search_start, len(full_ids) - m + 1):
                    positions = set(range(j, j + m))
                    if positions & used_positions:
                        continue
                    if full_ids[j:j + m] == ch_ids:
                        char_indices.extend(range(j, j + m))
                        search_start = j + m
                        break
            if char_indices:
                result[text] = char_indices
                used_positions.update(char_indices)
    return result


def generate_hard_negative(text: str) -> str | None:
    """Generate a confusable variant: one jamo substituted per syllable."""
    return substitute_per_char(text)


def render_text_image(
    text: str, base_height: int,
    font_path: str | None = None,
    align: int = 16, padding: int = 64,
) -> Image.Image:
    """Render bold black text on white, sized to fit the text content."""
    fs = max(base_height // 3, 48)
    if font_path:
        font = ImageFont.truetype(font_path, fs)
    else:
        font = ImageFont.load_default(size=fs)
    # measure actual text bounding box (includes ascender/descender offsets)
    tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=font)
    tw, th = x1 - x0, y1 - y0
    # image size: fit text + padding, align to 16px
    w = (tw + padding * 2 + align - 1) // align * align
    h = (th + padding * 2 + align - 1) // align * align
    w, h = max(align, w), max(align, h)
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)
    # center text accounting for bbox offset
    x = (w - tw) // 2 - x0
    y = (h - th) // 2 - y0
    sw = max(fs // 30, 1) # 수정전: sw = max(fs // 15 - 1, 1)
    draw.text((x, y), text, fill="black", font=font, stroke_width=sw, stroke_fill="black")
    return img


def compute_contrastive_loss(
    capture: "LayerInputCapture",
    layer_indices: list[int],
    synth_img_seq_len: int,
    temperature: float = 0.07,
    device: torch.device | None = None,
) -> torch.Tensor:
    """InfoNCE contrastive loss on DiT hidden states.

    Expects capture to hold states from a batch-2 forward where
    batch[0] = positive (correct text) rendering,
    batch[1] = negative (confusable text) rendering,
    both conditioned on the same text prompt.

    For each layer: pool image-token hidden states and text-token hidden states,
    then push sim(pos_img, txt) > sim(neg_img, txt) via cross-entropy.
    """
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0
    for layer_idx in layer_indices:
        x = capture.stored.get(f"x_{layer_idx}")
        if x is None or x.shape[0] < 2:
            continue
        # image tokens -> spatial mean pool
        pos_img_h = x[0, :synth_img_seq_len].mean(dim=0)
        neg_img_h = x[1, :synth_img_seq_len].mean(dim=0)
        # text tokens -> mean pool (same prompt for both, use batch 0)
        txt_h = x[0, synth_img_seq_len:].mean(dim=0)

        pos_img_h = F.normalize(pos_img_h.float(), dim=-1)
        neg_img_h = F.normalize(neg_img_h.float(), dim=-1)
        txt_h = F.normalize(txt_h.float(), dim=-1)

        pos_sim = (pos_img_h * txt_h).sum()
        neg_sim = (neg_img_h * txt_h).sum()
        logits = torch.stack([pos_sim, neg_sim]).unsqueeze(0) / temperature
        labels = torch.zeros(1, dtype=torch.long, device=device)
        total_loss = total_loss + F.cross_entropy(logits, labels)
        n_terms += 1

    return total_loss / max(n_terms, 1)


def log_layer_similarity_gaps(
    capture: "LayerInputCapture",
    layer_indices: list[int],
    synth_img_seq_len: int,
    global_step: int,
):
    """Log per-layer similarity gap (pos_sim - neg_sim) for contrastive layer selection.

    Layers with SMALL gap = model can't distinguish confusable pairs -> contrastive helps most.
    Layers with LARGE gap = already discriminative -> less benefit.
    """
    gaps = {}
    for layer_idx in layer_indices:
        x = capture.stored.get(f"x_{layer_idx}")
        if x is None or x.shape[0] < 2:
            continue
        with torch.no_grad():
            pos_img_h = F.normalize(x[0, :synth_img_seq_len].mean(dim=0).float(), dim=-1)
            neg_img_h = F.normalize(x[1, :synth_img_seq_len].mean(dim=0).float(), dim=-1)
            txt_h = F.normalize(x[0, synth_img_seq_len:].mean(dim=0).float(), dim=-1)
            pos_sim = (pos_img_h * txt_h).sum().item()
            neg_sim = (neg_img_h * txt_h).sum().item()
            gaps[layer_idx] = (pos_sim - neg_sim, pos_sim, neg_sim)

    gap_str = " | ".join(
        f"L{k}: gap={v[0]:+.4f} (pos={v[1]:.4f}, neg={v[2]:.4f})"
        for k, v in sorted(gaps.items())
    )
    logger.info(f"[layer diagnostic] step={global_step}\n{gap_str}")


class LayerInputCapture:
    """Capture inputs to selected transformer layers via forward pre-hooks.

    The captured tensors are checkpoint-boundary outputs (have grad_fn even
    under gradient checkpointing), so losses derived from them back-propagate
    correctly through the LoRA parameters.
    """

    def __init__(self):
        self.stored: dict[str, torch.Tensor] = {}
        self._hooks = []

    def register(self, transformer_module, layer_indices):
        for idx in layer_indices:
            layer = transformer_module.layers[idx]

            def _make_hook(layer_idx):
                def hook(module, args):
                    # args: (x, attn_mask, freqs_cis, adaln_input, ...)
                    self.stored[f"x_{layer_idx}"] = args[0]
                    self.stored[f"freqs_{layer_idx}"] = args[2]
                    if len(args) > 3:
                        self.stored[f"adaln_{layer_idx}"] = args[3]
                return hook

            self._hooks.append(layer.register_forward_pre_hook(_make_hook(idx)))

    def clear(self):
        self.stored.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def compute_char_loss(
    capture: LayerInputCapture,
    layer_indices: list[int],
    transformer_module,
    img_seq_len: int,
    batch_masks: list[dict],
    batch_token_indices: list[dict],
    device: torch.device,
):
    """Compute character localization loss from captured layer inputs.

    For each selected layer, Q and K are *recomputed* from the captured input
    (which sits on a checkpoint boundary and retains grad_fn).  This keeps the
    char-loss gradient flowing through to_q / to_k LoRA weights even when
    gradient checkpointing is enabled.

    L_char = -1/c * Σ_i (A_i · M_i  -  A_i · (1-M_i))
    """
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0

    for layer_idx in layer_indices:
        x = capture.stored.get(f"x_{layer_idx}")
        freqs_cis = capture.stored.get(f"freqs_{layer_idx}")
        adaln_input = capture.stored.get(f"adaln_{layer_idx}")
        if x is None:
            continue

        layer = transformer_module.layers[layer_idx]
        attn = layer.attention

        # --- recompute Q, K with AdaLN scaling, QK-norm, RoPE ---
        if layer.modulation and adaln_input is not None:
            mod = layer.adaLN_modulation(adaln_input)
            scale_msa = 1.0 + mod.unsqueeze(1).chunk(4, dim=2)[0]
            normed = layer.attention_norm1(x) * scale_msa
        else:
            normed = layer.attention_norm1(x)

        q = attn.to_q(normed)
        k = attn.to_k(normed)
        n_heads = attn.heads
        head_dim = q.shape[-1] // n_heads
        q = q.unflatten(-1, (n_heads, head_dim))
        k = k.unflatten(-1, (n_heads, head_dim))
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        if freqs_cis is not None:
            q = _apply_rotary_emb(q, freqs_cis)
            k = _apply_rotary_emb(k, freqs_cis)

        bsz = q.shape[0]
        for b in range(bsz):
            masks = batch_masks[b]
            tok_map = batch_token_indices[b]
            if not tok_map:
                continue

            # image queries / text keys — per-head attention then average
            q_img = q[b, :img_seq_len]                    # (img_seq, n_heads, head_dim)
            k_txt = k[b, img_seq_len:]                    # (txt_seq, n_heads, head_dim)
            if k_txt.shape[0] == 0:
                continue

            q_h = q_img.permute(1, 0, 2)                  # (n_heads, img_seq, head_dim)
            k_h = k_txt.permute(1, 2, 0)                  # (n_heads, head_dim, txt_seq)
            logits = torch.bmm(q_h, k_h) / (head_dim ** 0.5)
            attn_map = F.softmax(logits, dim=-1).mean(dim=0)  # (img_seq, txt_seq)

            for text, tok_indices in tok_map.items():
                if text not in masks:
                    continue
                mask_flat = masks[text].flatten()          # (img_seq,)
                if mask_flat.sum() == 0:
                    continue

                # attention to this word's tokens, averaged
                a_i = attn_map[:, tok_indices].mean(dim=-1)   # (img_seq,)

                # L = -(in_region - out_region)
                in_region = (a_i * mask_flat).sum()
                out_region = (a_i * (1.0 - mask_flat)).sum()
                total_loss = total_loss + -(in_region - out_region)
                n_terms += 1

    if n_terms > 0:
        total_loss = total_loss / n_terms
    return total_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir)),
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae_scale_factor = 8
    patch_size = 2

    # ---- Load models ----
    tokenizer = Qwen2Tokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    vae_shift = vae.config.shift_factor
    vae_scale = vae.config.scaling_factor

    transformer = ZImageTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)
    text_encoder = Qwen3Model.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder")

    vae.requires_grad_(False)
    transformer.requires_grad_(False)
    text_encoder.requires_grad_(False)

    vae.to(device=accelerator.device, dtype=weight_dtype)
    transformer.to(device=accelerator.device, dtype=weight_dtype)
    text_encoder.to(device=accelerator.device, dtype=weight_dtype)

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "w1", "w2", "w3"],
    )
    transformer.add_adapter(lora_config)
    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # ---- Text encoding pipeline (reuses text_encoder + tokenizer) ----
    text_encoding_pipeline = ZImagePipeline(
        vae=vae, transformer=transformer, tokenizer=tokenizer,
        text_encoder=text_encoder, scheduler=noise_scheduler,
    )

    def encode_prompts(prompts):
        """Encode prompts -> List[Tensor[seq_len, hidden_dim]] (no CFG)."""
        with torch.no_grad():
            prompt_embeds, _ = text_encoding_pipeline.encode_prompt(
                prompt=prompts,
                do_classifier_free_guidance=False,
                max_sequence_length=args.max_sequence_length,
            )
        return prompt_embeds

    # ---- Dataset & DataLoader ----
    train_dataset = ManifestDataset(args.manifest, args.max_pixels)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.dataloader_num_workers, drop_last=True,
    )
    logger.info(f"Dataset: {len(train_dataset):,} samples")

    # ---- Pre-cache latents & prompt embeddings ----
    # Caching must happen AFTER accelerator.prepare() to match dataloader order on multi-GPU.
    # We defer it to after prepare below.

    # ---- Optimizer ----
    trainable_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(trainable_params, lr=args.learning_rate, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, weight_decay=1e-4)

    # ---- LR scheduler ----
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # ---- Accelerate prepare ----
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if accelerator.is_main_process:
        accelerator.init_trackers("z-image-lora-simple", config=vars(args))
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Helpers ----
    def unwrap(model):
        m = accelerator.unwrap_model(model)
        return m._orig_mod if is_compiled_module(m) else m

    # ---- Char-loss setup ----
    use_char_loss = args.char_loss_lambda > 0
    char_loss_layers = [int(x) for x in args.char_loss_layers.split(",")] if use_char_loss else []
    latent_stride = vae_scale_factor * patch_size  # 16

    # ---- Contrastive loss setup ----
    use_contrastive = args.contrastive_lambda > 0
    contrastive_layers = [int(x) for x in args.contrastive_layers.split(",")] if use_contrastive else []
    synthetic_transform = None
    if use_contrastive:
        if not args.font_path:
            raise ValueError("--font_path is required when --contrastive_lambda > 0 (default font cannot render Korean)")
        synthetic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        logger.info(f"Contrastive loss enabled: lambda={args.contrastive_lambda}, "
                     f"layers={contrastive_layers}, temperature={args.contrastive_temperature}, "
                     f"synth_base_height={args.synthetic_size} (width adapts to text length)")

    # ---- Hook registration (union of char-loss + contrastive layers) ----
    capture = None
    all_capture_layers = sorted(set(char_loss_layers + contrastive_layers))
    if all_capture_layers:
        capture = LayerInputCapture()
        capture.register(unwrap(transformer), all_capture_layers)
    if use_char_loss:
        logger.info(f"Char-loss enabled: lambda={args.char_loss_lambda}, layers={char_loss_layers}, "
                     f"latent_stride={latent_stride} (per-image variable resolution)")

    # ---- Pre-cache latents & prompt embeddings (after prepare, using prepared dataloader) ----
    latents_cache = []
    prompt_embeds_cache = []
    masks_cache = []
    token_indices_cache = []
    if args.cache_latents:
        logger.info("Caching latents and prompt embeddings...")
        for batch in tqdm(train_dataloader, desc="Caching", disable=not accelerator.is_local_main_process):
            with torch.no_grad():
                pv = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                latents_cache.append(vae.encode(pv).latent_dist)
                prompt_embeds_cache.append(encode_prompts(batch["prompts"]))
            if use_char_loss:
                target_w, target_h = batch["target_sizes"][0]
                ph = target_h // latent_stride
                pw = target_w // latent_stride
                masks_cache.append(create_spatial_masks(
                    batch["bboxes_list"], batch["orig_sizes"],
                    target_h, target_w, ph, pw, accelerator.device))
                token_indices_cache.append([
                    find_text_token_indices(tokenizer, p, t, args.max_sequence_length)
                    for p, t in zip(batch["prompts"], batch["texts_list"])
                ])
        vae = vae.to("cpu")
        del vae
        text_encoding_pipeline.to("cpu")
        del text_encoder, tokenizer
        free_memory()
        logger.info(f"Cached {len(latents_cache)} batches")

    # ---- Save/Load hooks ----
    def save_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            lora_layers = get_peft_model_state_dict(unwrap(transformer))
            if weights:
                weights.pop()
            ZImagePipeline.save_lora_weights(
                output_dir, transformer_lora_layers=lora_layers,
                **_collate_lora_metadata({"transformer": unwrap(transformer)}))

    def load_hook(models, input_dir):
        while len(models) > 0:
            models.pop()
        lora_state = ZImagePipeline.lora_state_dict(input_dir)
        t_state = {k.replace("transformer.", ""): v for k, v in lora_state.items() if k.startswith("transformer.")}
        t_state = convert_unet_state_dict_to_peft(t_state)
        set_peft_model_state_dict(unwrap(transformer), t_state, adapter_name="default")
        if args.mixed_precision == "fp16":
            cast_training_params([unwrap(transformer)])

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    # ---- Resume ----
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path:
            accelerator.print(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)

    # ---- Sigma helper ----
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    # ---- Training loop ----
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                # Encode
                if args.cache_latents:
                    prompt_embeds = prompt_embeds_cache[step % len(prompt_embeds_cache)]
                    model_input = latents_cache[step % len(latents_cache)].mode()
                else:
                    with torch.no_grad():
                        prompt_embeds = encode_prompts(batch["prompts"])
                        pv = batch["pixel_values"].to(accelerator.device, dtype=vae.dtype)
                        model_input = vae.encode(pv).latent_dist.mode()

                model_input = (model_input - vae_shift) * vae_scale
                noise = torch.randn_like(model_input)
                bsz = model_input.shape[0]

                # Timestep sampling
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale)
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

                # Flow matching: noisy input
                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                timestep_normalized = (1000 - timesteps) / 1000

                # Z-Image transformer expects List[Tensor(C,1,H,W)]
                noisy_list = list(noisy_model_input.unsqueeze(2).unbind(dim=0))

                # Clear captured states before forward
                if capture is not None:
                    capture.clear()

                model_pred_list = transformer(
                    noisy_list, timestep_normalized, prompt_embeds, return_dict=False)[0]
                model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)
                model_pred = -model_pred  # Z-Image negates

                # MSE loss (全体)
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                loss_mse = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(bsz, -1), 1)
                loss_mse = loss_mse.mean()

                # Character localization loss
                char_loss = torch.tensor(0.0, device=accelerator.device)
                if use_char_loss:
                    if args.cache_latents:
                        batch_masks = masks_cache[step % len(masks_cache)]
                        batch_tok_idx = token_indices_cache[step % len(token_indices_cache)]
                        cur_img_seq_len = model_input.shape[-2] // patch_size * (model_input.shape[-1] // patch_size)
                    else:
                        target_w, target_h = batch["target_sizes"][0]
                        cur_patch_h = target_h // latent_stride
                        cur_patch_w = target_w // latent_stride
                        cur_img_seq_len = cur_patch_h * cur_patch_w
                        batch_masks = create_spatial_masks(
                            batch["bboxes_list"], batch["orig_sizes"],
                            target_h, target_w, cur_patch_h, cur_patch_w, accelerator.device)
                        batch_tok_idx = [
                            find_text_token_indices(tokenizer, p, t, args.max_sequence_length)
                            for p, t in zip(batch["prompts"], batch["texts_list"])
                        ]
                    char_loss = compute_char_loss(
                        capture, char_loss_layers, unwrap(transformer),
                        cur_img_seq_len, batch_masks, batch_tok_idx, accelerator.device)

                    # Debug: log char-loss details on first step
                    if global_step == 0 and accelerator.is_main_process:
                        n_masks = sum(len(m) for m in batch_masks)
                        n_matched = sum(len(t) for t in batch_tok_idx)
                        logger.info(f"[char-loss debug] masks={n_masks}, token_matches={n_matched}, "
                                     f"char_loss={char_loss.item():.6f}")

                # Contrastive glyph discriminability loss
                contrastive_loss = torch.tensor(0.0, device=accelerator.device)
                if use_contrastive:
                    # Pick one text randomly from the batch
                    b_idx = random.randint(0, bsz - 1)
                    if args.cache_latents:
                        texts_b = texts_cache[step % len(texts_cache)][b_idx]
                    else:
                        texts_b = batch["texts_list"][b_idx]
                    anchor = random.choice(texts_b)
                    neg_text = generate_hard_negative(anchor)

                    if neg_text is None and global_step == 0 and accelerator.is_main_process:
                        logger.warning(f"[contrastive] No confusable variant for '{anchor}', skipping")

                    if neg_text is not None:
                        # Render positive (GT text) and negative (confusable) images
                        pos_img = render_text_image(
                            anchor, args.synthetic_size, args.font_path)
                        neg_img = render_text_image(
                            neg_text, args.synthetic_size, args.font_path)
                        # Resize neg to match pos (same char count -> similar size, but ensure stack works)
                        if neg_img.size != pos_img.size:
                            neg_img = neg_img.resize(pos_img.size, Image.BILINEAR)

                        # Compute dynamic synth_img_seq_len from rendered size
                        pw, ph = pos_img.size  # PIL (w, h)
                        cur_synth_img_seq_len = (ph // latent_stride) * (pw // latent_stride)

                        synth_pv = torch.stack([
                            synthetic_transform(pos_img),
                            synthetic_transform(neg_img),
                        ]).to(accelerator.device, dtype=weight_dtype)

                        # VAE encode synthetic images
                        with torch.no_grad():
                            synth_latents = vae.encode(synth_pv).latent_dist.mode()
                        synth_latents = (synth_latents - vae_shift) * vae_scale

                        # Add noise at the same sigma as the main sample
                        synth_noise = torch.randn_like(synth_latents)
                        sigma_s = sigmas[b_idx].unsqueeze(0).expand(2, -1, -1, -1)
                        synth_noisy = (1.0 - sigma_s) * synth_latents + sigma_s * synth_noise

                        with torch.no_grad():
                            synth_embed = encode_prompts([f"The text '{anchor}'"])[0]
                        synth_prompt = [synth_embed, synth_embed]
                        synth_ts = timestep_normalized[b_idx:b_idx + 1].expand(2)
                        synth_noisy_list = list(synth_noisy.unsqueeze(2).unbind(dim=0))

                        # Diagnostic: wider hooks for layer selection (first 3 steps only)
                        diag_capture = None
                        if global_step < 3 and accelerator.is_main_process:
                            diag_layers = [i for i in range(0, 30, 2) if i not in contrastive_layers]
                            if diag_layers:
                                diag_capture = LayerInputCapture()
                                diag_capture.register(unwrap(transformer), diag_layers)

                        # Forward synthetic images through DiT
                        capture.clear()
                        _ = transformer(
                            synth_noisy_list, synth_ts, synth_prompt, return_dict=False)

                        contrastive_loss = compute_contrastive_loss(
                            capture, contrastive_layers, cur_synth_img_seq_len,
                            args.contrastive_temperature, accelerator.device)

                        # Log layer-wise similarity gaps then clean up
                        if diag_capture is not None:
                            all_diag = {**diag_capture.stored, **capture.stored}
                            combined = LayerInputCapture()
                            combined.stored = all_diag
                            all_layer_indices = sorted(
                                set([i for i in range(0, 30, 2)] + contrastive_layers))
                            log_layer_similarity_gaps(
                                combined, all_layer_indices, cur_synth_img_seq_len, global_step)
                            diag_capture.remove()

                        # Debug: save synthetic images & log on first few steps
                        if global_step < 3 and accelerator.is_main_process:
                            debug_dir = os.path.join(args.output_dir, "debug_synth")
                            os.makedirs(debug_dir, exist_ok=True)
                            pos_img.save(os.path.join(debug_dir, f"step{global_step}_pos_{anchor}.png"))
                            neg_img.save(os.path.join(debug_dir, f"step{global_step}_neg_{neg_text}.png"))
                            logger.info(
                                f"[contrastive debug] anchor='{anchor}', neg='{neg_text}', "
                                f"loss={contrastive_loss.item():.6f}, "
                                f"saved to {debug_dir}")

                char_loss = char_loss.clamp(min=0)
                loss = (loss_mse
                        + args.char_loss_lambda * char_loss
                        + args.contrastive_lambda * contrastive_loss)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]))
                        while len(ckpts) >= args.checkpoints_total_limit:
                            shutil.rmtree(os.path.join(args.output_dir, ckpts.pop(0)))
                    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    logger.info(f"Saved checkpoint-{global_step}")

            logs = {
                "loss": loss.detach().item(),
                "loss_mse": loss_mse.detach().item(),
                "loss_char": char_loss.detach().item() if use_char_loss else 0.0,
                "loss_contrastive": contrastive_loss.detach().item() if use_contrastive else 0.0,
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # Validation
        if accelerator.is_main_process and args.validation_prompt and epoch % args.validation_epochs == 0:
            pipeline = ZImagePipeline.from_pretrained(
                args.pretrained_model_name_or_path, transformer=unwrap(transformer), torch_dtype=weight_dtype)
            pipeline.enable_model_cpu_offload()
            gen = torch.Generator(device="cpu").manual_seed(args.seed) if args.seed else None
            images = [pipeline(prompt=args.validation_prompt, generator=gen).images[0]
                      for _ in range(args.num_validation_images)]
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    import numpy as np
                    tracker.writer.add_images("validation",
                        np.stack([np.asarray(img) for img in images]), epoch, dataformats="NHWC")
            del pipeline
            free_memory()

    # ---- Final save ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        t = unwrap(transformer).to(weight_dtype)
        lora_layers = get_peft_model_state_dict(t)
        ZImagePipeline.save_lora_weights(
            args.output_dir, transformer_lora_layers=lora_layers,
            **_collate_lora_metadata({"transformer": t}))
        logger.info(f"LoRA saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
