"""Inference for Z-Image LoRA + ControlNet (glyph conditioning) across checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from diffusers import ZImagePipeline
from diffusers.models.controlnets.controlnet_z_image import ZImageControlNetModel
from diffusers.pipelines.z_image.pipeline_z_image_controlnet import ZImageControlNetPipeline

FONT_PATH = str(Path(__file__).resolve().parent.parent / "NotoSansKR-VariableFont_wght.ttf")


def render_glyph_canvas(
    img_w: int, img_h: int,
    bboxes: list, texts: list,
    font_path: str = FONT_PATH,
) -> Image.Image:
    canvas = Image.new("RGB", (img_w, img_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for bbox, text in zip(bboxes, texts):
        x, y, bw, bh = bbox
        if bw < 4 or bh < 4 or not text:
            continue

        is_vertical = bh > bw

        if is_vertical:
            n_chars = len(text)
            char_h = bh / n_chars
            font_size = max(int(min(bw, char_h) * 0.85), 8)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except OSError:
                font = ImageFont.load_default()
            for i, ch in enumerate(text):
                bx0, by0, bx1, by1 = font.getbbox(ch)
                ch_w, ch_h = bx1 - bx0, by1 - by0
                cx = x + (bw - ch_w) / 2 - bx0
                cy = y + i * char_h + (char_h - ch_h) / 2 - by0
                draw.text((cx, cy), ch, fill=(255, 255, 255), font=font)
        else:
            font_size = max(int(bh * 0.85), 8)
            try:
                font = ImageFont.truetype(font_path, font_size)
            except OSError:
                font = ImageFont.load_default()
            bx0, by0, bx1, by1 = font.getbbox(text)
            text_w = bx1 - bx0
            if text_w > bw and text_w > 0:
                font_size = max(int(font_size * bw / text_w), 8)
                try:
                    font = ImageFont.truetype(font_path, font_size)
                except OSError:
                    font = ImageFont.load_default()
                bx0, by0, bx1, by1 = font.getbbox(text)
            text_w, text_h = bx1 - bx0, by1 - by0
            tx = x + (bw - text_w) / 2 - bx0
            ty = y + (bh - text_h) / 2 - by0
            draw.text((tx, ty), text, fill=(255, 255, 255), font=font)

    return canvas


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, required=True, help="Base Z-Image model path")
    p.add_argument("--training_dir", type=str, required=True,
                   help="Training output dir containing checkpoint-* folders and controlnet/")
    p.add_argument("--output_dir", type=str, default="controlnet_glyph_results")
    p.add_argument("--eval_jsonl", type=str, default=None,
                   help="Eval dataset jsonl with text, bbox, caption fields")
    p.add_argument("--prompt", type=str, nargs="*", default=None,
                   help="Manual prompts (used if --eval_jsonl not provided)")
    p.add_argument("--texts", type=str, nargs="*", default=None,
                   help="Texts to render (one per prompt, comma-separated for multiple)")
    p.add_argument("--bboxes", type=str, nargs="*", default=None,
                   help="Bboxes as 'x,y,w,h' (one per prompt)")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--conditioning_scale", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_sequence_length", type=int, default=512)
    p.add_argument("--font_path", type=str, default=FONT_PATH)
    return p.parse_args()


def find_checkpoints(training_dir: Path) -> list[Path]:
    ckpts = sorted(
        [d for d in training_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]),
    )
    return ckpts


def build_samples_from_jsonl(jsonl_path: str) -> list[dict]:
    samples = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            texts = rec["text"] if isinstance(rec["text"], list) else [rec["text"]]
            bbox_dict = rec.get("bbox", {})
            bboxes = [bbox_dict[k] for k in bbox_dict]
            bbox_texts = list(bbox_dict.keys())
            caption = rec.get("caption", "")
            text_str = ", ".join(f"'{t}'" for t in texts)
            prompt = f"{caption}, these texts are written on it: {text_str}" if caption else \
                     f"A signage photo, these texts are written on it: {text_str}"
            w = rec.get("width", 1024)
            h = rec.get("height", 1024)
            samples.append({
                "prompt": prompt,
                "bboxes": bboxes,
                "bbox_texts": bbox_texts,
                "width": w,
                "height": h,
            })
    return samples


def build_samples_from_args(args) -> list[dict]:
    samples = []
    prompts = args.prompt or []
    texts_list = args.texts or []
    bboxes_list = args.bboxes or []

    for i, prompt in enumerate(prompts):
        text = texts_list[i].split(",") if i < len(texts_list) else ["텍스트"]
        bbox_strs = bboxes_list[i].split(";") if i < len(bboxes_list) else []
        bboxes = []
        for bs in bbox_strs:
            parts = [int(x) for x in bs.split(",")]
            bboxes.append(parts)
        # Default bbox if not provided
        if not bboxes:
            bboxes = [[100, 400, 800, 200]]
        samples.append({
            "prompt": prompt,
            "bboxes": bboxes,
            "bbox_texts": text,
            "width": args.width,
            "height": args.height,
        })
    return samples


def main():
    args = parse_args()
    training_dir = Path(args.training_dir)
    output_dir = Path(args.output_dir)

    checkpoints = find_checkpoints(training_dir)
    final_weights = training_dir / "pytorch_lora_weights.safetensors"
    if final_weights.exists():
        checkpoints.append(training_dir)

    if not checkpoints:
        print(f"No checkpoints found in {training_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoint(s): {[c.name for c in checkpoints]}")

    # Build samples
    if args.eval_jsonl:
        samples = build_samples_from_jsonl(args.eval_jsonl)
    else:
        samples = build_samples_from_args(args)

    if not samples:
        print("No samples to generate")
        return

    # Load base pipeline
    base_pipe = ZImagePipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)

    for ckpt_path in checkpoints:
        ckpt_name = ckpt_path.name if ckpt_path != training_dir else "final"
        ckpt_output = output_dir / ckpt_name
        ckpt_output.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {ckpt_name} ===")

        # Load ControlNet
        cn_path = ckpt_path / "controlnet" if ckpt_path != training_dir else training_dir / "controlnet"
        controlnet = ZImageControlNetModel.from_pretrained(cn_path, torch_dtype=torch.bfloat16)
        controlnet = ZImageControlNetModel.from_transformer(controlnet, base_pipe.transformer)

        # Build ControlNet pipeline
        pipe = ZImageControlNetPipeline(
            vae=base_pipe.vae,
            transformer=base_pipe.transformer,
            controlnet=controlnet,
            tokenizer=base_pipe.tokenizer,
            text_encoder=base_pipe.text_encoder,
            scheduler=base_pipe.scheduler,
        )
        pipe.to("cuda")

        for i, sample in enumerate(samples):
            w, h = sample["width"], sample["height"]
            # Align to 16
            w = w // 16 * 16
            h = h // 16 * 16

            glyph = render_glyph_canvas(w, h, sample["bboxes"], sample["bbox_texts"], args.font_path)

            generator = torch.Generator(device="cuda").manual_seed(args.seed)
            image = pipe(
                prompt=sample["prompt"],
                control_image=glyph,
                height=h,
                width=w,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.conditioning_scale,
                generator=generator,
                max_sequence_length=args.max_sequence_length,
            ).images[0]

            out_path = ckpt_output / f"{i:04d}.png"
            image.save(out_path)

            # Save glyph for reference
            glyph_path = ckpt_output / f"{i:04d}_glyph.png"
            glyph.save(glyph_path)

            print(f"  [{i}] '{sample['prompt'][:60]}...' -> {out_path}")

        del pipe, controlnet
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
