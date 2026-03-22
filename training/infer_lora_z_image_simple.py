"""Inference for Z-Image LoRA across checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from diffusers import ZImagePipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Base Z-Image model path")
    p.add_argument("--training_dir", type=str, required=True,
                   help="Training output dir containing checkpoint-* folders")
    p.add_argument("--prompt", type=str, nargs="+", required=True)
    p.add_argument("--output_dir", type=str, default="lora_simple_results")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_sequence_length", type=int, default=512)
    return p.parse_args()


def find_checkpoints(training_dir: Path) -> list[Path]:
    ckpts = sorted(
        [d for d in training_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return ckpts


def main():
    args = parse_args()
    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir)

    checkpoints = find_checkpoints(training_dir)
    # Also check if final weights exist in training_dir root
    final_weights = training_dir / "pytorch_lora_weights.safetensors"
    if final_weights.exists():
        checkpoints.append(training_dir)

    if not checkpoints:
        print(f"No checkpoints found in {training_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoint(s): {[c.name for c in checkpoints]}")

    # Load base pipeline once
    pipe = ZImagePipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    for ckpt_path in checkpoints:
        ckpt_name = ckpt_path.name if ckpt_path != training_dir else "final"
        ckpt_output = output_dir / ckpt_name

        expected = {ckpt_output / f"{i:04d}.png" for i in range(len(args.prompt))}
        if expected and all(p.exists() for p in expected):
            print(f"\n=== {ckpt_name} === SKIP (already done)")
            continue

        ckpt_output.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {ckpt_name} ===")
        pipe.load_lora_weights(ckpt_path)

        for i, prompt in enumerate(args.prompt):
            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            image = pipe(
                prompt=prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                max_sequence_length=args.max_sequence_length,
            ).images[0]

            out_path = ckpt_output / f"{i:04d}.png"
            image.save(out_path)
            print(f"  [{i}] '{prompt}' -> {out_path}")

        pipe.unload_lora_weights()


if __name__ == "__main__":
    main()
