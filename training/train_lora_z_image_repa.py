"""Z-Image LoRA fine-tuning with bbox-masked loss + REPA-style OCR alignment."""

import argparse
import copy
import json
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm

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

logger = get_logger(__name__)

VAE_SCALE_FACTOR = 8
DIT_PATCH_SIZE = 2  # Z-Image patchifies latents with patch_size=2


# ---------------------------------------------------------------------------
# Projection MLP (REPA-style: DiT hidden → OCR feature dim)
# ---------------------------------------------------------------------------
class ProjectionMLP(nn.Module):
    def __init__(self, dit_dim: int, proj_dim: int, ocr_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dit_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.SiLU(),
            nn.Linear(proj_dim, ocr_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# OCR encoder wrapper (EasyOCR CRNN)
# ---------------------------------------------------------------------------
class OCREncoder:
    def __init__(self, lang: str = "ko", device: torch.device = torch.device("cpu")):
        import easyocr
        self.reader = easyocr.Reader([lang], gpu=(device.type == "cuda"), verbose=False)
        self.model = self.reader.recognizer.module
        self.model.eval()
        self.device = device
        self._seq_features = None
        self._hook = self.model.SequenceModeling.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, input, output):
        if isinstance(output, tuple):
            self._seq_features = output[0].detach()
        else:
            self._seq_features = output.detach()

    @property
    def embed_dim(self) -> int:
        return 256  # EasyOCR BiLSTM hidden_size * 2 directions / 2 = 256

    @torch.no_grad()
    def encode_crop(self, crop: Image.Image, target_h: int = 64) -> torch.Tensor:
        """Encode a single text crop → (1, T, D) sequence features."""
        w, h = crop.size
        ratio = target_h / h
        target_w = max(int(w * ratio), 1)
        img = crop.resize((target_w, target_h), Image.BILINEAR).convert("L")
        arr = np.array(img).astype("float32") / 255.0
        arr = (arr - 0.5) / 0.5
        tensor = torch.from_numpy(arr[np.newaxis, np.newaxis]).to(self.device)  # (1, 1, H, W)
        self._seq_features = None
        self.model(tensor, text="")
        return self._seq_features  # (1, T, 256)


# ---------------------------------------------------------------------------
# Bbox → DiT patch token indices
# ---------------------------------------------------------------------------
def bbox_to_dit_patch_indices(
    bbox: list, img_h: int, img_w: int,
) -> list[int]:
    """Map pixel-space COCO bbox [x,y,w,h] to DiT patch token indices.

    DiT patchifies the latent (lH, lW) with patch_size=2,
    giving a grid of (lH/2, lW/2) patches, flattened row-major.
    """
    lh = img_h // VAE_SCALE_FACTOR
    lw = img_w // VAE_SCALE_FACTOR
    grid_h = lh // DIT_PATCH_SIZE
    grid_w = lw // DIT_PATCH_SIZE

    x, y, bw, bh = bbox
    # Convert pixel bbox to patch grid coordinates
    px0 = max(0, int(x) // (VAE_SCALE_FACTOR * DIT_PATCH_SIZE))
    py0 = max(0, int(y) // (VAE_SCALE_FACTOR * DIT_PATCH_SIZE))
    px1 = min(grid_w, math.ceil((x + bw) / (VAE_SCALE_FACTOR * DIT_PATCH_SIZE)))
    py1 = min(grid_h, math.ceil((y + bh) / (VAE_SCALE_FACTOR * DIT_PATCH_SIZE)))

    indices = []
    for r in range(py0, py1):
        for c in range(px0, px1):
            indices.append(r * grid_w + c)
    return indices


# ---------------------------------------------------------------------------
# REPA alignment loss
# ---------------------------------------------------------------------------
def compute_repa_loss(
    dit_hidden: torch.Tensor,       # (N, T_img + T_txt, D) from hook
    ocr_features: torch.Tensor,     # (1, T_ocr, ocr_dim)
    patch_indices: list[int],        # which dit image patches correspond to bbox
    projector: ProjectionMLP,
    img_seq_len: int,                # number of image tokens in dit_hidden
) -> torch.Tensor:
    """Compute REPA-style alignment loss between DiT bbox patches and OCR features."""
    if len(patch_indices) == 0:
        return torch.tensor(0.0, device=dit_hidden.device)

    # Extract image-only tokens, then select bbox patches
    img_tokens = dit_hidden[0, :img_seq_len]  # (T_img, D)
    idx = torch.tensor(patch_indices, device=img_tokens.device)
    bbox_tokens = img_tokens[idx]  # (N_patches, D)

    # Project to OCR dimension
    projected = projector(bbox_tokens.float())  # (N_patches, ocr_dim)

    # Pool OCR features to match patch count (simple interpolation)
    ocr_feat = ocr_features[0]  # (T_ocr, ocr_dim)
    # Interpolate OCR sequence to match bbox patch count
    ocr_interp = F.interpolate(
        ocr_feat.unsqueeze(0).permute(0, 2, 1),  # (1, D, T_ocr)
        size=len(patch_indices),
        mode="linear",
        align_corners=False,
    ).permute(0, 2, 1).squeeze(0)  # (N_patches, ocr_dim)

    # Cosine alignment (REPA-style: maximize cosine similarity)
    projected_n = F.normalize(projected, dim=-1)
    ocr_n = F.normalize(ocr_interp.detach().float(), dim=-1)
    loss = -(projected_n * ocr_n).sum(dim=-1).mean()  # negative cosine similarity

    return loss


# ---------------------------------------------------------------------------
# Args, dataset, mask (same as simple_masked)
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="z-image-lora-repa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pixels", type=int, default=1024*1024)
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
    p.add_argument("--use_8bit_adam", action="store_true")
    # REPA-specific args
    p.add_argument("--repa_coeff", type=float, default=0.5,
                   help="Weight for REPA alignment loss")
    p.add_argument("--repa_layer", type=int, default=8,
                   help="DiT layer index to extract hidden states from (0-indexed)")
    p.add_argument("--proj_dim", type=int, default=1024,
                   help="Projection MLP intermediate dimension")
    return p.parse_args()


def build_prompt(caption: str, texts: list) -> str:
    text_str = ", ".join(texts)
    if caption:
        return f"{caption}, texts are written on it: {text_str}"
    return f"A signage photo, texts are written on it: {text_str}"


class ManifestDataset(Dataset):
    ALIGN = 16

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
        orig_w, orig_h = image.size
        if image.mode != "RGB":
            image = image.convert("RGB")

        new_w, new_h = self._fit_size(orig_w, orig_h)
        if (new_w, new_h) != (orig_w, orig_h):
            image = image.resize((new_w, new_h), Image.BILINEAR)

        sx, sy = new_w / orig_w, new_h / orig_h
        bboxes = []
        texts_for_bbox = []
        bbox_dict = rec.get("bbox", {})
        for key in bbox_dict:
            x, y, w, h = bbox_dict[key]
            bboxes.append([x * sx, y * sy, w * sx, h * sy])
            texts_for_bbox.append(key)

        texts = rec["text"] if isinstance(rec["text"], list) else [rec["text"]]
        return {
            "pixel_values": self.to_tensor(image),
            "image": image,  # keep PIL for OCR crop
            "prompt": build_prompt(rec.get("caption", ""), texts),
            "bboxes": bboxes,
            "bbox_texts": texts_for_bbox,
            "img_size": (new_h, new_w),
        }


def build_latent_mask(bboxes: list, img_size: tuple, device: torch.device) -> torch.Tensor:
    h, w = img_size
    lh, lw = h // VAE_SCALE_FACTOR, w // VAE_SCALE_FACTOR
    mask = torch.zeros(1, 1, lh, lw, device=device)
    for x, y, bw, bh in bboxes:
        lx0 = max(0, int(x) // VAE_SCALE_FACTOR)
        ly0 = max(0, int(y) // VAE_SCALE_FACTOR)
        lx1 = min(lw, int(x + bw + VAE_SCALE_FACTOR - 1) // VAE_SCALE_FACTOR)
        ly1 = min(lh, int(y + bh + VAE_SCALE_FACTOR - 1) // VAE_SCALE_FACTOR)
        mask[:, :, ly0:ly1, lx0:lx1] = 1.0
    return mask


def collate_fn(examples):
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {
        "pixel_values": pixel_values,
        "images": [e["image"] for e in examples],
        "prompts": [e["prompt"] for e in examples],
        "bboxes": [e["bboxes"] for e in examples],
        "bbox_texts": [e["bbox_texts"] for e in examples],
        "img_sizes": [e["img_size"] for e in examples],
    }


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
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)],
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

    # ---- OCR encoder (frozen) ----
    ocr_encoder = OCREncoder(lang="ko", device=accelerator.device)
    logger.info(f"OCR encoder loaded: EasyOCR CRNN, embed_dim={ocr_encoder.embed_dim}")

    # ---- Projection MLP ----
    dit_dim = transformer.config["dim"]  # 3840
    projector = ProjectionMLP(dit_dim, args.proj_dim, ocr_encoder.embed_dim)
    projector.to(accelerator.device, dtype=torch.float32)
    logger.info(f"Projector: {dit_dim} → {args.proj_dim} → {ocr_encoder.embed_dim}")

    # ---- DiT hidden state hook ----
    dit_hidden_state = {}

    def make_dit_hook(layer_idx):
        def hook_fn(module, input, output):
            dit_hidden_state[layer_idx] = output
        return hook_fn

    hook_handle = transformer.layers[args.repa_layer].register_forward_hook(
        make_dit_hook(args.repa_layer))
    logger.info(f"REPA hook registered at layer {args.repa_layer}/{len(transformer.layers)}")

    # ---- Text encoding pipeline ----
    text_encoding_pipeline = ZImagePipeline(
        vae=vae, transformer=transformer, tokenizer=tokenizer,
        text_encoder=text_encoder, scheduler=noise_scheduler,
    )

    def encode_prompts(prompts):
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

    # ---- Optimizer (LoRA params + projector params) ----
    lora_params = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    proj_params = list(projector.parameters())
    all_trainable = lora_params + proj_params
    logger.info(f"Trainable params: {sum(p.numel() for p in all_trainable):,} "
                f"(LoRA: {sum(p.numel() for p in lora_params):,}, "
                f"Projector: {sum(p.numel() for p in proj_params):,})")

    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(all_trainable, lr=args.learning_rate, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(all_trainable, lr=args.learning_rate, weight_decay=1e-4)

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
    transformer, projector, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, projector, optimizer, train_dataloader, lr_scheduler)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if accelerator.is_main_process:
        accelerator.init_trackers("z-image-lora-repa", config=vars(args))
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Save/Load hooks ----
    def unwrap(model):
        m = accelerator.unwrap_model(model)
        return m._orig_mod if is_compiled_module(m) else m

    def save_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            lora_layers = get_peft_model_state_dict(unwrap(transformer))
            if weights:
                weights.pop()
            ZImagePipeline.save_lora_weights(
                output_dir, transformer_lora_layers=lora_layers,
                **_collate_lora_metadata({"transformer": unwrap(transformer)}))
            # Save projector separately
            torch.save(unwrap(projector).state_dict(),
                       os.path.join(output_dir, "projector.pt"))

    def load_hook(models, input_dir):
        while len(models) > 0:
            models.pop()
        lora_state = ZImagePipeline.lora_state_dict(input_dir)
        t_state = {k.replace("transformer.", ""): v for k, v in lora_state.items() if k.startswith("transformer.")}
        t_state = convert_unet_state_dict_to_peft(t_state)
        set_peft_model_state_dict(unwrap(transformer), t_state, adapter_name="default")
        if args.mixed_precision == "fp16":
            cast_training_params([unwrap(transformer)])
        proj_path = os.path.join(input_dir, "projector.pt")
        if os.path.exists(proj_path):
            unwrap(projector).load_state_dict(torch.load(proj_path, map_location="cpu"))

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
        projector.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer, projector):
                # Encode
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

                sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise
                timestep_normalized = (1000 - timesteps) / 1000

                # Z-Image transformer expects List[Tensor(C,1,H,W)]
                noisy_list = list(noisy_model_input.unsqueeze(2).unbind(dim=0))

                # Clear hook state
                dit_hidden_state.clear()

                model_pred_list = transformer(
                    noisy_list, timestep_normalized, prompt_embeds, return_dict=False)[0]
                model_pred = torch.stack(model_pred_list, dim=0).squeeze(2)
                model_pred = -model_pred

                # ---- Masked MSE loss (same as simple_masked) ----
                masks = torch.cat([
                    build_latent_mask(batch["bboxes"][i], batch["img_sizes"][i], model_input.device)
                    for i in range(bsz)
                ], dim=0)

                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                target = noise - model_input
                per_pixel_loss = weighting.float() * (model_pred.float() - target.float()) ** 2
                masked_loss = per_pixel_loss * masks.float()
                n_channels = model_input.shape[1]
                mask_sums = masks.reshape(bsz, -1).sum(dim=1) * n_channels
                mask_sums = mask_sums.clamp(min=1.0)
                loss_mse = (masked_loss.reshape(bsz, -1).sum(dim=1) / mask_sums).mean()

                # ---- REPA alignment loss ----
                loss_repa = torch.tensor(0.0, device=accelerator.device)
                repa_count = 0

                if args.repa_layer in dit_hidden_state and args.repa_coeff > 0:
                    hidden = dit_hidden_state[args.repa_layer]
                    # hidden shape: (bsz, T_img + T_txt, dim) — joint attention tokens
                    for i in range(bsz):
                        img_h, img_w = batch["img_sizes"][i]
                        lh = img_h // VAE_SCALE_FACTOR
                        lw = img_w // VAE_SCALE_FACTOR
                        img_seq_len = (lh // DIT_PATCH_SIZE) * (lw // DIT_PATCH_SIZE)

                        for j, bbox in enumerate(batch["bboxes"][i]):
                            patch_idx = bbox_to_dit_patch_indices(bbox, img_h, img_w)
                            if len(patch_idx) == 0:
                                continue

                            # Crop text region from PIL image for OCR
                            pil_img = batch["images"][i]
                            x, y, bw, bh = bbox
                            x, y, bw, bh = int(x), int(y), int(bw), int(bh)
                            crop = pil_img.crop((x, y, x + bw, y + bh))
                            if crop.size[0] < 4 or crop.size[1] < 4:
                                continue

                            ocr_feat = ocr_encoder.encode_crop(crop)  # (1, T_ocr, 256)
                            if ocr_feat is None:
                                continue

                            h_i = hidden[i].unsqueeze(0) if hidden.dim() == 3 else hidden[i:i+1]
                            repa_l = compute_repa_loss(
                                h_i, ocr_feat, patch_idx, projector, img_seq_len)
                            loss_repa = loss_repa + repa_l
                            repa_count += 1

                if repa_count > 0:
                    loss_repa = loss_repa / repa_count

                # ---- Total loss ----
                loss = loss_mse + args.repa_coeff * loss_repa

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(all_trainable, args.max_grad_norm)
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
                "loss_repa": loss_repa.detach().item() if isinstance(loss_repa, torch.Tensor) else 0.0,
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
        torch.save(unwrap(projector).state_dict(),
                   os.path.join(args.output_dir, "projector.pt"))
        logger.info(f"LoRA + projector saved to {args.output_dir}")

    hook_handle.remove()
    accelerator.end_training()


if __name__ == "__main__":
    main()
