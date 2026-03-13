"""Batch inference for Z-Image LoRA evaluation.

Reads eval dataset jsonl, generates images using LoRA checkpoint,
saves results for downstream evaluation (OCR accuracy, CLIPScore, FID).
"""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from diffusers import ZImagePipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Base Z-Image model path")
    p.add_argument("--lora_path", type=str, default=None, help="LoRA weights path (dir or checkpoint-*)")
    p.add_argument("--eval_jsonl", type=str, required=True, help="Eval dataset jsonl")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for generated images")
    p.add_argument("--num_samples", type=int, default=4, help="Number of samples per prompt")
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--max_eval", type=int, default=None, help="Limit number of eval samples")
    return p.parse_args()


def load_eval_dataset(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_prompt(rec: dict) -> str:
    caption = rec.get("caption", "")
    texts = rec.get("text", [])
    if isinstance(texts, str):
        texts = [texts]
    text_str = ", ".join(f"'{t}'" for t in texts)
    if caption:
        return f"{caption}, with {text_str} written on it"
    return f"A signage photo, with {text_str} written on it"


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_eval_dataset(args.eval_jsonl)
    if args.max_eval:
        records = records[:args.max_eval]
    print(f"Loaded {len(records)} eval samples")

    pipe = ZImagePipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    pipe.to("cuda")

    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)
        print(f"Loaded LoRA from {args.lora_path}")

    # Use GT dimensions from dataset
    for idx, rec in enumerate(tqdm(records, desc="Generating")):
        stem = Path(rec["image_path"]).stem
        prompt = build_prompt(rec)
        h = rec.get("height", 1024)
        w = rec.get("width", 1024)

        for s in range(args.num_samples):
            generator = torch.Generator(device="cuda").manual_seed(args.seed + s)
            image = pipe(
                prompt=prompt,
                height=h,
                width=w,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                max_sequence_length=args.max_sequence_length,
            ).images[0]

            out_path = output_dir / f"{stem}_{s}.jpg"
            image.save(out_path)

    # Save generation metadata
    meta = {
        "model_path": args.model_path,
        "lora_path": args.lora_path,
        "num_samples": args.num_samples,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "num_records": len(records),
    }
    (output_dir / "gen_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Done: {len(records) * args.num_samples} images -> {output_dir}")


if __name__ == "__main__":
    main()
