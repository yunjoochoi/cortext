"""Inference for Kandinsky 2.2 inpaint LoRA checkpoints."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from peft import LoraConfig

from diffusers import KandinskyV22InpaintPipeline, KandinskyV22PriorPipeline, UNet2DConditionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROMPTS = [
    {
        "prompt": "A signage image containing Korean text '안녕하세요'.",
        "negative_prompt": "",
        "height": 1024,
        "width": 1024,
        "seed": 42,
        "sample_steps": 10,
        "cfg_scale": 5.0,
    },
    {
        "prompt": "A signage image containing Korean text '커피숍'.",
        "negative_prompt": "",
        "height": 1024,
        "width": 1024,
        "seed": 42,
        "sample_steps": 10,
        "cfg_scale": 5.0,
    },
    {
        "prompt": "A signage image containing Korean text '닭볶음탕'.",
        "negative_prompt": "",
        "height": 1024,
        "width": 1024,
        "seed": 42,
        "sample_steps": 10,
        "cfg_scale": 5.0,
    },
]


def find_checkpoints(training_dir: Path) -> list[Path]:
    """Find checkpoint-* dirs sorted by step number."""
    ckpts = sorted(
        [d for d in training_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return ckpts


def load_lora_from_checkpoint(
    base_model_path: str,
    ckpt_path: Path,
    lora_rank: int,
    weight_dtype: torch.dtype,
    device: str,
) -> UNet2DConditionModel:
    """Load base UNet, add LoRA adapter, and load checkpoint weights."""
    unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    # accelerator.save_state saves full model (base + LoRA) in model.safetensors
    ckpt_file = ckpt_path / "model.safetensors"
    if not ckpt_file.exists():
        raise FileNotFoundError(f"No model.safetensors in {ckpt_path}")

    state_dict = load_file(str(ckpt_file))

    # Filter to only LoRA keys and load them
    lora_keys = {k: v for k, v in state_dict.items() if "lora" in k}
    logger.info(f"Loading {len(lora_keys)} LoRA keys from {ckpt_path.name}")
    unet.load_state_dict(state_dict, strict=False)

    unet.to(device, dtype=weight_dtype)
    unet.eval()
    return unet


def run_inference(
    prior_pipe: KandinskyV22PriorPipeline,
    inpaint_pipe: KandinskyV22InpaintPipeline,
    prompts: list[dict],
    output_dir: Path,
    device: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, p in enumerate(prompts):
        h, w = p["height"], p["width"]
        generator = torch.Generator(device=device).manual_seed(p["seed"])

        # Prior: text → image_embeds
        image_emb, zero_image_emb = prior_pipe(
            p["prompt"],
            negative_prompt=p["negative_prompt"] or None,
            num_inference_steps=25,
            generator=generator,
        ).to_tuple()

        # Blank image + full mask → generate from scratch
        init_image = Image.new("RGB", (w, h), (255, 255, 255))
        mask = np.ones((h, w), dtype=np.float32)

        generator = torch.Generator(device=device).manual_seed(p["seed"])
        result = inpaint_pipe(
            image=init_image,
            mask_image=mask,
            image_embeds=image_emb,
            negative_image_embeds=zero_image_emb,
            height=h,
            width=w,
            num_inference_steps=p["sample_steps"],
            guidance_scale=p["cfg_scale"],
            generator=generator,
        )

        safe_text = p["prompt"].split("'")[1] if "'" in p["prompt"] else f"prompt_{i}"
        out_path = output_dir / f"{i:02d}_{safe_text}.png"
        result.images[0].save(out_path)
        logger.info(f"Saved: {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference for Kandinsky 2.2 inpaint LoRA.")
    p.add_argument("--training_dir", type=str, required=True,
                    help="Directory with checkpoint-* subdirs.")
    p.add_argument("--decoder_model", type=str, default="kandinsky-community/kandinsky-2-2-decoder-inpaint")
    p.add_argument("--prior_model", type=str, default="kandinsky-community/kandinsky-2-2-prior")
    p.add_argument("--output_dir", type=str, default=None, help="Defaults to <training_dir>/inference/")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    p.add_argument("--baseline", action="store_true", help="Also generate baseline (no LoRA) images.")
    return p.parse_args()


def main():
    args = parse_args()
    training_dir = Path(args.training_dir)
    output_base = Path(args.output_dir) if args.output_dir else training_dir / "inference"

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    weight_dtype = dtype_map[args.dtype]
    device = "cuda"

    # Load prior (shared across all checkpoints)
    logger.info("Loading prior pipeline...")
    prior_pipe = KandinskyV22PriorPipeline.from_pretrained(
        args.prior_model, torch_dtype=weight_dtype
    ).to(device)

    # Load base inpaint pipeline
    logger.info("Loading inpaint decoder pipeline...")
    inpaint_pipe = KandinskyV22InpaintPipeline.from_pretrained(
        args.decoder_model, torch_dtype=weight_dtype
    ).to(device)

    # Baseline
    if args.baseline:
        logger.info("=== Baseline (no LoRA) ===")
        run_inference(prior_pipe, inpaint_pipe, PROMPTS, output_base / "baseline", device)

    # Iterate checkpoints
    ckpts = find_checkpoints(training_dir)
    logger.info(f"Found {len(ckpts)} checkpoint(s): {[c.name for c in ckpts]}")

    for ckpt_dir in ckpts:
        logger.info(f"=== {ckpt_dir.name} ===")

        unet = load_lora_from_checkpoint(
            args.decoder_model, ckpt_dir, args.lora_rank, weight_dtype, device
        )
        inpaint_pipe.unet = unet

        run_inference(prior_pipe, inpaint_pipe, PROMPTS, output_base / ckpt_dir.name, device)

        del unet
        torch.cuda.empty_cache()

    logger.info("Done.")


if __name__ == "__main__":
    main()
