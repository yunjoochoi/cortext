"""Fix remaining quoted text in clean captions: regex strip + Qwen3 rephrase."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from tqdm import tqdm
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

from core.utils import read_jsonl, write_jsonl

QWEN_MODEL = "/scratch2/shaush/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
QUOTED_RE = re.compile(r'\s*["\u201C\u201D]([^"\u201C\u201D]{2,})["\u201C\u201D]\s*')

SYSTEM_PROMPT = (
    "You are a grammar editor. You receive a caption that has some awkward phrasing "
    "where text was removed. Rewrite it to be grammatically natural. "
    "Do NOT add new information. Keep the same style and similar length."
)

USER_PROMPT_TEMPLATE = (
    "{stripped}\n\n"
    "Rewrite this caption to be grammatically natural. Just fix the grammar, "
    "do not add any new words or details that weren't there.\n\n"
    "Examples of fixes:\n"
    '- "A bright yellow sign for is mounted on the facade" -> "A bright yellow sign is mounted on the facade"\n'
    '- "A storefront with a logo and a phone number" -> "A storefront with a logo and a phone number"\n'
    '- "featuring and on a red background" -> "featuring text on a red background"\n'
    "CLEAN: <rewritten caption>"
)


def strip_quotes(text: str) -> str:
    result = QUOTED_RE.sub(" ", text)
    result = re.sub(r"  +", " ", result).strip()
    return result


def parse_clean(text: str, fallback: str) -> str:
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("CLEAN:"):
            return line[len("CLEAN:"):].strip()
    return fallback


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Directory with *_clean.jsonl files")
    p.add_argument("--pattern", default="captions_shard_*_clean.jsonl")
    p.add_argument("--model", default=QWEN_MODEL)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--dry_run", action="store_true", help="Print regex results without model")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    clean_files = sorted(input_dir.glob(args.pattern))
    if not clean_files:
        print(f"No files matching {args.pattern} in {input_dir}")
        sys.exit(1)

    to_fix: list[tuple[Path, int, dict, str]] = []
    for fpath in clean_files:
        for idx, line in enumerate(open(fpath)):
            row = json.loads(line)
            cc = row.get("clean_caption", "")
            if QUOTED_RE.search(cc):
                stripped = strip_quotes(cc)
                to_fix.append((fpath, idx, row, stripped))

    print(f"Found {len(to_fix)} captions with quoted text across {len(clean_files)} files")
    if not to_fix:
        return

    if args.dry_run:
        for fpath, idx, row, stripped in to_fix:
            print(f"\n[{fpath.name}:L{idx+1}]")
            print(f"  BEFORE:   {row['clean_caption'][:150]}")
            print(f"  STRIPPED: {stripped[:150]}")
        return

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    processor.tokenizer.padding_side = "left"

    fixes: dict[tuple[Path, int], str] = {}

    for i in tqdm(range(0, len(to_fix), args.batch_size), desc="Fixing quoted"):
        batch = to_fix[i : i + args.batch_size]
        messages_batch = []
        for _, _, _, stripped in batch:
            messages_batch.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(stripped=stripped)},
            ])

        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for m in messages_batch
        ]
        inputs = processor(text=texts, padding=True, return_tensors="pt").to(model.device)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)

        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        for (fpath, idx, row, stripped), ids in zip(batch, trimmed):
            response = processor.decode(ids, skip_special_tokens=True).strip()
            clean = parse_clean(response, stripped)
            fixes[(fpath, idx)] = clean
            print(f"  [{fpath.name}:L{idx+1}]")
            print(f"    BEFORE:   {row['clean_caption'][:120]}")
            print(f"    STRIPPED: {stripped[:120]}")
            print(f"    AFTER:    {clean[:120]}")

    for fpath in clean_files:
        lines = open(fpath).readlines()
        changed = False
        for idx in range(len(lines)):
            if (fpath, idx) in fixes:
                row = json.loads(lines[idx])
                row["clean_caption"] = fixes[(fpath, idx)]
                row["was_fixed"] = True
                lines[idx] = json.dumps(row, ensure_ascii=False) + "\n"
                changed = True
        if changed:
            n = sum(1 for idx in range(len(lines)) if (fpath, idx) in fixes)
            with open(fpath, "w") as f:
                f.writelines(lines)
            print(f"\nUpdated {n} lines in {fpath.name}")

    print(f"\nDone: {len(fixes)} captions fixed")


if __name__ == "__main__":
    main()
