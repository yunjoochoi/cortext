"""Detect and remove leaked text from VLM captions using Qwen3-VL (text-only)."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from core.utils import read_jsonl, write_jsonl

QWEN_MODEL = "/scratch2/shaush/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

SYSTEM_PROMPT = (
    "You are a caption editor. Remove any text that was read from signs, banners, or labels. "
    "This includes ANY language (Korean, English, Chinese, etc.), numbers, brand names, and symbols."
)

USER_PROMPT_TEMPLATE = (
    'Caption: "{caption}"\n\n'
    'Step 1: List any text READ from signs — any language, numbers, brand names, phone numbers.\n'
    'Text in quotes (e.g. "OPEN", "Barber Shop") is almost always read from a sign.\n'
    '"a barber shop exterior" = OK (scene description).\n'
    '"a sign reading Barber Shop" / "the English words Barber shop" = leaked.\n'
    'Step 2: If leaked text found, rewrite without it. Keep same style and length.\n'
    'If none found, return caption as-is.\n\n'
    'LEAKED: <list, or "none">\n'
    'CLEAN: <cleaned caption>'
)


def parse_response(text: str, original: str) -> tuple[str, str]:
    """Parse model response into (leaked_text, clean_caption)."""
    leaked = "none"
    clean = original
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("LEAKED:"):
            leaked = line[len("LEAKED:"):].strip()
        elif line.upper().startswith("CLEAN:"):
            clean = line[len("CLEAN:"):].strip()
    return leaked, clean


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Caption shard jsonl (with vlm_caption field)")
    p.add_argument("--output", default=None, help="Output jsonl (default: input with _clean suffix)")
    p.add_argument("--model", default=QWEN_MODEL)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None)
    args = p.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_stem(input_path.stem + "_clean")

    records = list(read_jsonl(input_path))
    if args.max_samples:
        records = records[:args.max_samples]
    print(f"Loaded {len(records):,} records from {input_path}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    processor.tokenizer.padding_side = "left"

    results = []
    fixed_count = 0

    for i in tqdm(range(0, len(records), args.batch_size), desc="Verifying captions"):
        batch = records[i : i + args.batch_size]
        messages_batch = []
        for rec in batch:
            caption = rec.get("vlm_caption", "")
            messages_batch.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(caption=caption)},
            ])

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for m in messages_batch
        ]
        inputs = processor(text=texts, padding=True, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        for rec, ids in zip(batch, trimmed):
            response = processor.decode(ids, skip_special_tokens=True).strip()
            leaked, clean = parse_response(response, rec.get("vlm_caption", ""))
            was_fixed = leaked.lower() != "none"
            if was_fixed:
                fixed_count += 1

            results.append({
                "image_path": rec["image_path"],
                "vlm_caption": rec.get("vlm_caption", ""),
                "leaked": leaked,
                "clean_caption": clean,
                "was_fixed": was_fixed,
            })

            if len(results) <= 20:
                status = "FIXED" if was_fixed else "OK"
                print(f"  [{status}] {leaked} -> {clean[:80]}")

    write_jsonl(output_path, results)
    print(f"\nDone: {fixed_count:,}/{len(results):,} captions fixed -> {output_path}")


if __name__ == "__main__":
    main()
