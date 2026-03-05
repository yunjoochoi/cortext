"""Inference for Z-Image LoRA (simple, no inpainting)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from diffusers import ZImagePipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Base Z-Image model path")
    p.add_argument("--lora_weights", type=str, required=True, help="LoRA weights dir (from save_lora_weights)")
    p.add_argument("--prompt", type=str, nargs="+", required=True)
    p.add_argument("--output_dir", type=str, default="lora_simple_results")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_sequence_length", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = ZImagePipeline.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    pipe.load_lora_weights(args.lora_weights)
    pipe.to("cuda")

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    for i, prompt in enumerate(args.prompt):
        image = pipe(
            prompt=prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            max_sequence_length=args.max_sequence_length,
        ).images[0]

        out_path = output_dir / f"{i:04d}.png"
        image.save(out_path)
        print(f"[{i}] '{prompt}' -> {out_path}")


if __name__ == "__main__":
    main()
