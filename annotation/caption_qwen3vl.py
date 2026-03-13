"""Generate 1-line captions for manifest images using Qwen3-VL (multi-GPU sharded)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from core.utils import read_jsonl, write_jsonl

QWEN_MODEL = "/scratch2/shaush/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

SYSTEM_PROMPT = (
    "You are a image captioner. "
    "Describe the image in one English sentence focusing on the scene, signage type, and surroundings. "
    "Do NOT mention or transcribe any specific text, letters, words, or characters visible in the image. "
    "Do NOT read the signs. Only describe the visual scene."
)

USER_PROMPT_TEMPLATE = (
    "This image contains the following text on signs: {texts}. "
    "Ignore all text content and describe only the visual scene in one sentence "
    "(building, street, objects, colors, lighting). Do NOT mention any text."
)

USER_PROMPT_NO_TEXT = (
    "Describe this image in one sentence. "
    "Focus only on the scene (building, street, objects, colors, lighting). "
    "Do NOT include any text or words that appear in the image."
)


def deduplicate_by_image(records: list[dict]) -> dict[str, dict]:
    """Return {image_path: record} keeping first occurrence per image."""
    seen = {}
    for rec in records:
        if rec["image_path"] not in seen:
            seen[rec["image_path"]] = rec
    return seen


def collect_texts_by_image(records: list[dict]) -> dict[str, list[str]]:
    """Collect all unique text strings per image across all records."""
    texts_map: dict[str, list[str]] = {}
    for rec in records:
        img = rec["image_path"]
        raw = rec.get("text", [])
        words = raw if isinstance(raw, list) else [raw]
        if img not in texts_map:
            texts_map[img] = []
        for w in words:
            if w and w not in texts_map[img]:
                texts_map[img].append(w)
    return texts_map


def build_final_caption(caption: str, texts: list[str]) -> str:
    """Combine VLM caption with text references: '{caption}, with text {t1} and {t2}'."""
    if not texts:
        return caption
    if len(texts) == 1:
        return f"{caption}, with text '{texts[0]}' written on it"
    quoted = [f"'{t}'" for t in texts]
    return f"{caption}, with {', '.join(quoted[:-1])} and {quoted[-1]} written on it"


def generate_captions(
    manifest_path: Path,
    output_path: Path,
    model_name: str = QWEN_MODEL,
    batch_size: int = 4,
    max_samples: int | None = None,
    rank: int = 0,
    world_size: int = 1,
):
    records = list(read_jsonl(manifest_path))
    unique_map = deduplicate_by_image(records)
    texts_map = collect_texts_by_image(records)
    unique_records = list(unique_map.values())
    if max_samples:
        unique_records = unique_records[:max_samples]

    # Shard: each rank processes its slice
    unique_records = unique_records[rank::world_size]
    print(f"  Rank {rank}/{world_size}: {len(unique_records):,} images (total unique: {len(unique_map):,})")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    processor.tokenizer.padding_side = "left"

    captions = {}
    for i in tqdm(range(0, len(unique_records), batch_size), desc=f"Captioning [rank {rank}]"):
        batch_recs = unique_records[i : i + batch_size]
        messages_batch = []
        images_batch = []

        for rec in batch_recs:
            img = Image.open(rec["image_path"]).convert("RGB")
            img.thumbnail((1024, 1024))
            images_batch.append(img)
            texts = len(texts_map.get(rec["image_path"], []))
            if texts:
                user_text = USER_PROMPT_NO_TEXT #USER_PROMPT_TEMPLATE.format(texts=texts)
            else:
                user_text = USER_PROMPT_NO_TEXT
            messages_batch.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": user_text},
                ]},
            ])

        texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False) for m in messages_batch]
        inputs = processor(
            text=texts, images=images_batch, padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=150, do_sample=False)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        for rec, ids in zip(batch_recs, trimmed):
            caption = processor.decode(ids, skip_special_tokens=True).strip()
            captions[rec["image_path"]] = caption

        if len(captions) <= 30:
            for rec in batch_recs:
                if len(captions) > 30:
                    break
                img_texts = texts_map.get(rec["image_path"], [])
                final = build_final_caption(captions[rec["image_path"]], img_texts)
                print(f"  [{len(captions)}] {final}")

    # Write shard output (raw VLM captions as jsonl for merging)
    shard_path = output_path.parent / f"captions_shard_{rank}.jsonl"
    shard_records = [{"image_path": k, "vlm_caption": v} for k, v in captions.items()]
    write_jsonl(shard_path, shard_records)
    print(f"  Rank {rank}: {len(captions):,} captions -> {shard_path}")

    # If single GPU, also write the final merged output directly
    if world_size == 1:
        _merge_and_write(records, texts_map, {0: shard_path}, output_path)


def merge_shards(
    manifest_path: Path,
    output_path: Path,
    world_size: int,
):
    """Merge shard files into final captioned manifest."""
    records = list(read_jsonl(manifest_path))
    texts_map = collect_texts_by_image(records)
    shard_paths = {i: output_path.parent / f"captions_shard_{i}.jsonl" for i in range(world_size)}
    _merge_and_write(records, texts_map, shard_paths, output_path)


def _merge_and_write(
    records: list[dict],
    texts_map: dict[str, list[str]],
    shard_paths: dict[int, Path],
    output_path: Path,
):
    """Load all shards, combine with manifest, write final output."""
    captions = {}
    for rank, path in shard_paths.items():
        for rec in read_jsonl(path):
            captions[rec["image_path"]] = rec["vlm_caption"]
    print(f"  Loaded {len(captions):,} captions from {len(shard_paths)} shard(s)")

    output_records = []
    for rec in records:
        img_path = rec["image_path"]
        if img_path in captions:
            rec["caption"] = build_final_caption(captions[img_path], texts_map.get(img_path, []))
        output_records.append(rec)

    write_jsonl(output_path, output_records)
    captioned = sum(1 for r in output_records if "caption" in r)
    print(f"  {captioned:,}/{len(output_records):,} records captioned -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/scratch2/shaush/coreset_output/manifest.jsonl")
    p.add_argument("--output", default="/scratch2/shaush/coreset_output/manifest_captioned.jsonl")
    p.add_argument("--model", default=QWEN_MODEL)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--merge_only", action="store_true", help="Only merge existing shards")
    args = p.parse_args()

    if args.merge_only:
        merge_shards(Path(args.manifest), Path(args.output), args.world_size)
    else:
        generate_captions(
            Path(args.manifest), Path(args.output),
            model_name=args.model, batch_size=args.batch_size,
            max_samples=args.max_samples,
            rank=args.rank, world_size=args.world_size,
        )