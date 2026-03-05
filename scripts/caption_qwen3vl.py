"""Add caption field to [라벨]Training JSON files using Qwen3-VL-8B-Instruct."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


CATEGORY_KO_TO_EN = {
    "1.간판": "signboard",
    "1.가로형간판": "horizontal signboard",
    "2.돌출간판": "protruding signboard",
    "3.세로형간판": "vertical signboard",
    "4.실내간판": "indoor signboard",
    "5.실내안내판": "indoor information board",
    "6.지주이용간판": "pole-mounted signboard",
    "7.창문이용광고물": "window advertisement",
    "8.현수막": "banner",
    "2.책표지": "book cover",
    "01.총류": "general works",
    "02.철학": "philosophy",
    "03.종교": "religion",
    "04.사회과학": "social sciences",
    "05.자연과학": "natural sciences",
    "06.기술과학": "technology",
    "07.예술": "arts",
    "08.언어": "language",
    "09.문학": "literature",
    "10.역사": "history",
    "11.기타": "miscellaneous",
}


def category_to_english(category: str) -> str:
    parts = category.split("/")
    return " / ".join(CATEGORY_KO_TO_EN.get(p, p) for p in parts)


def build_prompt(category_en: str) -> str:
    return (
        f"This is a photo of a {category_en} in South Korea. "
        "Describe the scene in one concise English sentence, focusing on the overall appearance, "
        "setting, and style. Do NOT transcribe or mention the specific text content."
    )


def build_image_lookup(source_roots: list[Path]) -> dict[str, str]:
    lookup = {}
    for root in source_roots:
        for img in root.rglob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                lookup[img.name] = str(img)
    return lookup


def load_model(model_path: str):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def caption_batch(
    image_paths: list[str],
    category_ens: list[str],
    model,
    processor,
) -> list[str]:
    all_messages = []
    for img_path, cat_en in zip(image_paths, category_ens):
        all_messages.append([{
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{img_path}"},
                {"type": "text", "text": build_prompt(cat_en)},
            ],
        }])

    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in all_messages
    ]
    all_images = []
    for m in all_messages:
        imgs, _ = process_vision_info(m)
        all_images.extend(imgs)

    inputs = processor(
        text=texts, images=all_images, padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

    pad_id = processor.tokenizer.pad_token_id
    captions = []
    for i in range(len(texts)):
        input_len = inputs.input_ids[i].ne(pad_id).sum().item()
        generated = output_ids[i][input_len:]
        captions.append(processor.decode(generated, skip_special_tokens=True).strip())
    return captions


def extract_category(json_path: Path, label_root: Path) -> str:
    relative = json_path.relative_to(label_root)
    return "/".join(relative.parts[:-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="/scratch2/shaush/030.야외_실제_촬영_한글_이미지")
    parser.add_argument("--label-subdir", type=str, default="[라벨]Training")
    parser.add_argument("--model-path", type=str,
                        default="/scratch2/shaush/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
    parser.add_argument("--cache-path", type=str, default="/scratch2/shaush/coreset_output/caption_cache.json",
                        help="Caption cache for resume support")
    parser.add_argument("--category-filter", type=str, default=None, help="e.g. '1.간판' to process only signboards")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=200)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    label_root = data_root / args.label_subdir

    # Find all label JSONs
    scan_root = label_root / args.category_filter if args.category_filter else label_root
    json_files = sorted(scan_root.rglob("*.json"))
    print(f"{len(json_files):,} label files under {scan_root}")

    # Build image lookup
    source_roots = sorted(data_root.glob("[원천]Training_*"))
    print(f"{len(source_roots)} source directories")
    image_lookup = build_image_lookup(source_roots)
    print(f"{len(image_lookup):,} images indexed")

    # Load caption cache
    cache_path = Path(args.cache_path)
    caption_cache: dict[str, str] = {}
    if cache_path.exists():
        caption_cache = json.loads(cache_path.read_text())
        print(f"Resumed {len(caption_cache):,} cached captions")

    # Collect pending items
    pending = []  # (json_path, file_name, image_path, category_en)
    skipped = 0
    for json_path in json_files:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        if "caption" in data:
            skipped += 1
            continue
        file_name = data["images"][0]["file_name"]
        if file_name in caption_cache:
            data["caption"] = caption_cache[file_name]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            continue
        image_path = image_lookup.get(file_name)
        if image_path is None:
            continue
        category_en = category_to_english(extract_category(json_path, label_root))
        pending.append((json_path, file_name, image_path, category_en))

    print(f"{len(pending):,} to caption, {skipped:,} already done")
    if not pending:
        return

    # Load model
    print("Loading Qwen3-VL...")
    model, processor = load_model(args.model_path)
    print("Model loaded.")

    captioned = 0
    bs = args.batch_size

    for i in range(0, len(pending), bs):
        batch = pending[i : i + bs]
        captions = caption_batch(
            [b[2] for b in batch], [b[3] for b in batch], model, processor,
        )
        for (json_path, file_name, _, _), caption in zip(batch, captions):
            caption_cache[file_name] = caption
            with open(json_path, encoding="utf-8") as f:
                data = json.load(f)
            data["caption"] = caption
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        captioned += len(batch)
        if captioned % args.save_every < bs:
            cache_path.write_text(json.dumps(caption_cache, ensure_ascii=False))
            print(f"[{captioned:,}/{len(pending):,}] {captions[-1][:80]}...")

    cache_path.write_text(json.dumps(caption_cache, ensure_ascii=False))
    print(f"Done. captioned={captioned:,}, cache={len(caption_cache):,}")


if __name__ == "__main__":
    main()
