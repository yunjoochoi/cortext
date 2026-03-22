"""Refine annotation bboxes into precise polygons via Qwen3-VL.

Input:  manifest JSONL (has text + bbox per annotation)
Output: JSONL with polygon added to each annotation, keyed by image_id
"""

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.utils import read_jsonl

QWEN_MODEL = "/scratch2/shaush/models/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"

SYSTEM_PROMPT = (
    "You are a precise text polygon annotator for Korean signage images. "
    "Given an image and a list of text regions with approximate bounding boxes, "
    "output a refined 4-point polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] for each region "
    "that tightly fits the actual text boundaries (including any slant or rotation). "
    "Points should be in clockwise order starting from top-left. "
    "Output ONLY a JSON array of N polygons in the same order as input. No explanation."
)


def build_user_prompt(annotations: list[dict], img_w: int, img_h: int) -> str:
    lines = [f"Image size: {img_w}x{img_h}px. Refine these {len(annotations)} text regions:\n"]
    for i, ann in enumerate(annotations):
        x, y, w, h = ann["bbox"]
        x2, y2 = x + w, y + h
        lines.append(f"{i+1}. Text: '{ann['text']}' — approx bbox: [{int(x)},{int(y)},{int(x2)},{int(y2)}]")
    lines.append(
        "\nFor each region, output a tight 4-point polygon [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] in pixel coords. "
        "Return a JSON array of exactly {n} polygons.".format(n=len(annotations))
    )
    return "\n".join(lines)


_JSON_ARR = re.compile(r'\[\s*\[.*?\]\s*\]', re.DOTALL)
_COORD_ROW = re.compile(r'\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]')


def parse_polygons(response: str, n: int, fallback_anns: list[dict]) -> list[list[list[float]]]:
    """Parse N polygons from VLM response. Falls back to bbox-derived polygon on failure."""
    def bbox_to_poly(ann):
        x, y, w, h = ann["bbox"]
        return [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]

    # Try to find outermost JSON array containing arrays of [x,y] pairs
    try:
        # Find all [...] blocks that look like polygon arrays
        m = re.search(r'(\[\s*\[\s*\[.*?\]\s*\]\s*\])', response, re.DOTALL)
        if m:
            polys = json.loads(m.group(1))
            if len(polys) == n and all(len(p) >= 3 for p in polys):
                return [[[float(pt[0]), float(pt[1])] for pt in p] for p in polys]
    except (json.JSONDecodeError, ValueError, IndexError):
        pass

    # Fallback: collect all [x,y] pairs and group into chunks of 4
    pairs = [[float(m.group(1)), float(m.group(2))] for m in _COORD_ROW.finditer(response)]
    if len(pairs) == n * 4:
        return [pairs[i*4:(i+1)*4] for i in range(n)]

    # Can't parse — use bbox fallback for all
    return [bbox_to_poly(ann) for ann in fallback_anns]


def refine_polygons(
    manifest_path: Path,
    output_path: Path,
    model_name: str,
    rank: int,
    world_size: int,
):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen3-VL on {device} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=device
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)

    records = [r for r in read_jsonl(manifest_path) if r.get("annotations")]
    records = records[rank::world_size]
    print(f"  Rank {rank}: {len(records):,} images")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path = output_path.with_suffix(f".shard{rank}.jsonl")

    with open(shard_path, "w", encoding="utf-8") as f:
        for rec in tqdm(records, desc=f"Rank {rank}"):
            anns = rec["annotations"]
            try:
                img = Image.open(rec["image_path"]).convert("RGB")
                img_w, img_h = img.size
            except Exception as e:
                print(f"  Warning: {rec.get('image_id')}: {e}")
                # fallback: bbox → polygon
                out_anns = [{**a, "polygon": [[a["bbox"][0], a["bbox"][1]],
                                              [a["bbox"][0]+a["bbox"][2], a["bbox"][1]],
                                              [a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]],
                                              [a["bbox"][0], a["bbox"][1]+a["bbox"][3]]]} for a in anns]
                f.write(json.dumps({"image_id": rec["image_id"], "annotations": out_anns},
                                   ensure_ascii=False) + "\n")
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": build_user_prompt(anns, img_w, img_h)},
                ]},
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            try:
                inputs = processor(text=[text], images=[img], return_tensors="pt").to(device)
                with torch.no_grad():
                    out_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                resp = processor.decode(
                    out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )
                polygons = parse_polygons(resp, len(anns), anns)
            except Exception as e:
                print(f"  Inference error {rec.get('image_id')}: {e}")
                polygons = [[[a["bbox"][0], a["bbox"][1]],
                             [a["bbox"][0]+a["bbox"][2], a["bbox"][1]],
                             [a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]],
                             [a["bbox"][0], a["bbox"][1]+a["bbox"][3]]] for a in anns]

            out_anns = [{**a, "polygon": poly} for a, poly in zip(anns, polygons)]
            f.write(json.dumps({"image_id": rec["image_id"], "annotations": out_anns},
                               ensure_ascii=False) + "\n")

    print(f"  Shard saved: {shard_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="/scratch2/shaush/coreset_output/manifest.jsonl")
    parser.add_argument("--output", default="/scratch2/shaush/coreset_output/polygon_refined_vlm.jsonl")
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()

    refine_polygons(
        Path(args.manifest), Path(args.output),
        args.model, args.rank, args.world_size,
    )
