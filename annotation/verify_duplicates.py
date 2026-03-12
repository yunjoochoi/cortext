"""Pick the best annotation among duplicate images using VLM verification."""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from core.utils import read_jsonl, write_jsonl

QWEN_MODEL = "/scratch2/shaush/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
FONT_PATH = str(Path(__file__).resolve().parents[1] / "NotoSansKR-VariableFont_wght.ttf")

BBOX_COLOR = "green"

SYSTEM_PROMPT = (
    "You are a Korean signage annotation quality checker. "
    "You will see an image with numbered bounding boxes drawn on it. "
    "Each annotation option lists the texts and their bounding boxes. "
    "Your job is to pick the single best annotation option, or reject all if none are acceptable."
)

JUDGE_PROMPT_TEMPLATE = """\
Below are {n} annotation options for the same image. Each option has a list of texts and bounding boxes.
The colored/numbered rectangles on the image correspond to each option's bounding boxes.

{options_block}

IMPORTANT: Only Korean text is annotated. Numbers, English, and other non-Korean characters are intentionally excluded from annotations and bboxes. Do NOT reject an option just because numbers or English text on the sign are not annotated.

Pick the BEST annotation option (1-indexed) based on these criteria IN ORDER:
0) OCR accuracy: the Korean texts must correctly match what is actually written in the image. If none are correct, return 0.
1) Bbox quality: each bbox must cover its corresponding Korean text without cutting through characters. Minor looseness is acceptable. If the best option has severely misaligned bboxes, return 0.
2) If multiple options are both correct, prefer the one with MORE text elements and longer text content.
3) Prefer single spaces over double spaces ("  " -> " "), and fewer spaces overall.
4) Prefer texts listed in natural reading order (top-to-bottom, left-to-right).

Reply with ONLY a single integer: the 1-indexed option number, or 0 if none are acceptable.
Do not explain."""


def md5_hash(filepath: str) -> str:
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def draw_bboxes_on_image(
    img: Image.Image, bboxes: dict[str, list], color: str, label_prefix: str,
) -> Image.Image:
    """Draw labeled bboxes on a copy of the image. bbox format: [x, y, w, h]."""
    img = img.copy()
    draw = ImageDraw.Draw(img)
    fs = max(img.height // 25, 24)
    lw = max(fs // 8, 2)
    try:
        font = ImageFont.truetype(FONT_PATH, fs)
    except OSError:
        font = ImageFont.load_default(size=fs)

    for i, (text, box) in enumerate(bboxes.items()):
        x, y, w, h = box
        draw.rectangle([x, y, x + w, y + h], outline=color, width=lw)
        label = f"{label_prefix}-{i}: {text}"
        draw.text((x + 2, max(0, y - fs - 4)), label, fill=color, font=font)
    return img


def build_options_block(records: list[dict]) -> str:
    lines = []
    for i, rec in enumerate(records, 1):
        texts = rec.get("text", [])
        bbox = rec.get("bbox", {})
        bbox_str = ", ".join(f'"{k}": {v}' for k, v in bbox.items())
        lines.append(f"Option {i}: texts={texts}, bboxes={{{bbox_str}}}")
    return "\n".join(lines)


def build_annotated_images(
    base_img: Image.Image, records: list[dict],
) -> list[Image.Image]:
    """Return one annotated image per option, each with its bboxes drawn."""
    images = []
    for i, rec in enumerate(records):
        annotated = draw_bboxes_on_image(base_img, rec.get("bbox", {}), BBOX_COLOR, f"Opt{i+1}")
        images.append(annotated)
    return images


def verify_duplicates(
    manifest_path: Path,
    output_path: Path,
    model_name: str = QWEN_MODEL,
    max_groups: int | None = None,
):
    records = list(read_jsonl(manifest_path))
    print(f"  Loaded {len(records):,} records")

    # Step 1: group by image hash
    print("  Computing image hashes...")
    hash_to_records: dict[str, list[dict]] = defaultdict(list)
    for rec in tqdm(records, desc="Hashing"):
        img_path = rec["image_path"]
        if not Path(img_path).exists():
            continue
        h = md5_hash(img_path)
        rec["_hash"] = h
        hash_to_records[h].append(rec)

    dup_groups = {h: recs for h, recs in hash_to_records.items() if len(recs) > 1}
    unique_records = [recs[0] for h, recs in hash_to_records.items() if len(recs) == 1]
    print(f"  {len(unique_records):,} unique, {len(dup_groups):,} duplicate groups ({sum(len(v) for v in dup_groups.values()):,} records)")

    if not dup_groups:
        print("  No duplicates found. Copying manifest as-is.")
        write_jsonl(output_path, records)
        return

    # Check if any group actually has differing annotations
    groups_to_judge = {}
    for h, recs in dup_groups.items():
        ann_keys = set(tuple(sorted(r.get("text", []))) for r in recs)
        if len(ann_keys) == 1:
            # All annotations identical — just keep first
            unique_records.append(recs[0])
        else:
            groups_to_judge[h] = recs
    print(f"  {len(groups_to_judge):,} groups with differing annotations need VLM judgment")

    if not groups_to_judge:
        print("  All duplicate annotations are identical. Writing deduplicated manifest.")
        _write_output(unique_records, output_path)
        return

    if max_groups:
        groups_to_judge = dict(list(groups_to_judge.items())[:max_groups])

    # Step 2: load model
    print(f"  Loading {model_name}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    # Step 3: judge each group
    debug_dir = output_path.parent / "verify_debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_count = 0
    chosen_records = []
    rejected_count = 0
    judgments = []  # metadata for post-processing

    for h, recs in tqdm(groups_to_judge.items(), desc="Judging"):
        base_img = Image.open(recs[0]["image_path"]).convert("RGB")
        annotated_images = build_annotated_images(base_img, recs)
        options_block = build_options_block(recs)

        user_prompt = JUDGE_PROMPT_TEMPLATE.format(n=len(recs), options_block=options_block)

        content_parts = []
        for ann_img in annotated_images:
            thumb = ann_img.copy()
            thumb.thumbnail((1024, 1024))
            content_parts.append({"type": "image", "image": thumb})
        content_parts.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content_parts},
        ]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = processor(
            text=[text_input],
            images=[img for img in annotated_images],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=16, do_sample=False)

        trimmed = output_ids[0][len(inputs.input_ids[0]):]
        answer = processor.decode(trimmed, skip_special_tokens=True).strip()

        try:
            choice = int(answer)
        except ValueError:
            # Try to extract first integer from response
            import re
            nums = re.findall(r"\d+", answer)
            choice = int(nums[0]) if nums else 0

        rejected = choice == 0 or choice > len(recs)
        for j, ann_img in enumerate(annotated_images):
            if rejected:
                tag = "_REJECT"
            elif j + 1 == choice:
                tag = "_CHOSEN"
            else:
                tag = ""
            name = f"group{debug_count:04d}_opt{j+1:02d}{tag}.jpg"
            ann_img.save(debug_dir / name)
        debug_count += 1

        judgment = {
            "hash": h,
            "choice": choice,
            "paths": [r["image_path"] for r in recs],
            "keep": None,
            "remove": [r["image_path"] for r in recs],
        }
        if choice == 0 or choice > len(recs):
            rejected_count += 1
            print(f"  Rejected group {h[:8]}... ({len(recs)} options, answer={answer})")
        else:
            winner = recs[choice - 1]
            chosen_records.append(winner)
            judgment["keep"] = winner["image_path"]
            judgment["remove"] = [p for p in judgment["paths"] if p != winner["image_path"]]
            print(f"  Group {h[:8]}...: picked option {choice}/{len(recs)}")
        judgments.append(judgment)

    print(f"\n  VLM judged {len(groups_to_judge)} groups: {len(chosen_records)} chosen, {rejected_count} rejected")

    # Save judgment metadata
    judgment_path = output_path.parent / "verify_judgments.jsonl"
    write_jsonl(judgment_path, judgments)
    print(f"  Judgment metadata -> {judgment_path}")

    final_records = unique_records + chosen_records
    _write_output(final_records, output_path)


def _write_output(records: list[dict], output_path: Path):
    # Remove internal keys
    clean = []
    for rec in records:
        r = {k: v for k, v in rec.items() if not k.startswith("_")}
        clean.append(r)
    write_jsonl(output_path, clean)
    print(f"  Wrote {len(clean):,} records -> {output_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="/scratch2/shaush/coreset_output/manifest.jsonl")
    p.add_argument("--output", default="/scratch2/shaush/coreset_output/manifest_verified.jsonl")
    p.add_argument("--model", default=QWEN_MODEL)
    p.add_argument("--max_groups", type=int, default=None, help="Limit number of groups to judge (for testing)")
    args = p.parse_args()

    verify_duplicates(
        Path(args.manifest), Path(args.output),
        model_name=args.model, max_groups=args.max_groups,
    )
