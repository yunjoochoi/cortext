"""Detect and remove leaked text from VLM captions using Qwen3-VL (text-only)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from core.utils import read_jsonl, write_jsonl

QWEN_MODEL = "/scratch2/shaush/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

SYSTEM_PROMPT = (
    "You are a caption editor. You MUST remove ALL text that was read or transcribed from signs, "
    "banners, labels, menus, or any visible surface in the image. This includes:\n"
    "- ANY language (Korean, English, Chinese, Japanese, etc.)\n"
    "- Brand names, store names, business names\n"
    "- Phone numbers, addresses, street names\n"
    "- Numbers read from signs (floor numbers like 2F, 3F, prices, percentages)\n"
    "- Menu items, product names\n"
    "- ANY text inside quotation marks — this is almost always transcribed from a sign\n\n"
    "Keep generic scene descriptions (e.g. 'a restaurant sign', 'a storefront'). "
    "Only remove the specific text/words that were read from the sign."
)

USER_PROMPT_TEMPLATE = (
    'Caption: "{caption}"\n\n'
    "Identify ALL leaked text — any word, name, number, or phrase that was READ from a sign or surface.\n"
    'Text in quotes (e.g. "OPEN", "Barber Shop", "2F") is almost always leaked.\n'
    'Romanized Korean (e.g. "Gwangma Gongjaksu") is leaked.\n'
    'Translated menu items (e.g. "Kimchi Stew") are leaked.\n'
    '"a barber shop exterior" = OK (scene description).\n'
    '"a sign reading Barber Shop" / "the English words Barber Shop" = leaked.\n\n'
    "If leaked text found, rewrite the caption without it. Keep same style and length.\n"
    "If none found, return caption as-is.\n\n"
    'LEAKED: <list, or "none">\n'
    "CLEAN: <cleaned caption>"
)


def parse_response(text: str, original: str) -> tuple[str, str]:
    leaked = "none"
    clean = original
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("LEAKED:"):
            leaked = line[len("LEAKED:"):].strip()
        elif line.upper().startswith("CLEAN:"):
            clean = line[len("CLEAN:"):].strip()
    return leaked, clean


def process_shard(
    input_path: Path,
    output_path: Path,
    model: Qwen3VLForConditionalGeneration,
    processor: AutoProcessor,
    batch_size: int,
) -> tuple[int, int]:
    records = list(read_jsonl(input_path))
    print(f"\n{'='*60}")
    print(f"Processing {input_path.name} ({len(records):,} records)")
    print(f"  -> {output_path.name}")
    print(f"{'='*60}")

    results = []
    fixed_count = 0

    for i in tqdm(range(0, len(records), batch_size), desc=input_path.stem):
        batch = records[i : i + batch_size]
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

            if len(results) <= 10:
                status = "FIXED" if was_fixed else "OK"
                print(f"  [{status}] {leaked} -> {clean[:80]}")

    write_jsonl(output_path, results)
    print(f"  Done: {fixed_count:,}/{len(results):,} fixed -> {output_path}")
    return len(results), fixed_count


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Directory containing captions_shard_*.jsonl")
    p.add_argument("--pattern", default="captions_shard_[0-9].jsonl", help="Glob pattern for shard files")
    p.add_argument("--model", default=QWEN_MODEL)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing _clean files")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    shard_files = sorted(input_dir.glob(args.pattern))
    if not shard_files:
        print(f"No files matching {args.pattern} in {input_dir}")
        sys.exit(1)

    print(f"Found {len(shard_files)} shards: {[f.name for f in shard_files]}")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    processor.tokenizer.padding_side = "left"

    total_records = 0
    total_fixed = 0

    for shard_path in shard_files:
        output_path = shard_path.with_stem(shard_path.stem + "_clean")
        if output_path.exists() and not args.overwrite:
            print(f"\nSkipping {shard_path.name} (_clean already exists, use --overwrite to redo)")
            continue

        n_records, n_fixed = process_shard(shard_path, output_path, model, processor, args.batch_size)
        total_records += n_records
        total_fixed += n_fixed

    print(f"\n{'='*60}")
    print(f"All done: {total_fixed:,}/{total_records:,} captions fixed across {len(shard_files)} shards")


if __name__ == "__main__":
    main()
