"""Generate 1-line captions for manifest images using Qwen3-VL."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from core.utils import read_jsonl, write_jsonl

QWEN_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

SYSTEM_PROMPT = "You are a concise image captioner. Describe the image in one short English sentence focusing on the scene, signage type, and surroundings. Do NOT mention specific text content."


def deduplicate_by_image(records: list[dict]) -> list[dict]:
    """Keep one record per unique image_path (avoid captioning the same image multiple times)."""
    seen = set()
    unique = []
    for rec in records:
        if rec["image_path"] not in seen:
            seen.add(rec["image_path"])
            unique.append(rec)
    return unique


def generate_captions(
    manifest_path: Path,
    output_path: Path,
    model_name: str = QWEN_MODEL,
    batch_size: int = 4,
    max_samples: int | None = None,
):
    records = list(read_jsonl(manifest_path))
    unique_records = deduplicate_by_image(records)
    if max_samples:
        unique_records = unique_records[:max_samples]
    print(f"  {len(records):,} total records, {len(unique_records):,} unique images to caption")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)

    captions = {}
    for i in tqdm(range(0, len(unique_records), batch_size), desc="Captioning"):
        batch_recs = unique_records[i : i + batch_size]
        messages_batch = []
        images_batch = []

        for rec in batch_recs:
            img = Image.open(rec["image_path"]).convert("RGB")
            img.thumbnail((1024, 1024))
            images_batch.append(img)
            messages_batch.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ]},
            ])

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_batch]
        inputs = processor(
            text=texts, images=images_batch, padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False)

        input_len = inputs.input_ids.shape[1]
        for rec, out_ids in zip(batch_recs, output_ids):
            caption = processor.decode(out_ids[input_len:], skip_special_tokens=True).strip()
            captions[rec["image_path"]] = caption

        if i == 0:
            print(f"  Sample caption: {captions[batch_recs[0]['image_path']]}")

    # Write back: attach caption to all records sharing the same image_path
    output_records = []
    for rec in records:
        if rec["image_path"] in captions:
            rec["caption"] = captions[rec["image_path"]]
        output_records.append(rec)

    write_jsonl(output_path, output_records)
    captioned = sum(1 for r in output_records if "caption" in r)
    print(f"  {captioned:,}/{len(output_records):,} records captioned -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/scratch2/shaush/coreset_output/manifest_selected.jsonl")
    p.add_argument("--output", default="/scratch2/shaush/coreset_output/manifest_captioned.jsonl")
    p.add_argument("--model", default=QWEN_MODEL)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    generate_captions(
        Path(args.manifest), Path(args.output),
        model_name=args.model, batch_size=args.batch_size,
        max_samples=args.max_samples,
    )
