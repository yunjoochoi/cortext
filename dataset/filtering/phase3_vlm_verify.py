"""
Phase 3: VLM annotation verification.

Input: Phase 1b + Phase 2 results merged (only surviving images/annotations)
Output: Per-annotation correct/reject/pending + image quality

Usage:
    python phase3_vlm_verify.py \
        --phase1b results/phase1b_vlm_dedup_1.jsonl results/phase1b_vlm_dedup_2.jsonl \
        --phase2 results/phase2_annotation_filter_XXXX.jsonl \
        --model_name Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json
import os

import cv2
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


# Load model
def load_model(model_name: str):
    print(f"Loading model: {model_name}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print("Model loaded", flush=True)
    return model, processor


# VLM call
def call_vlm(model, processor, messages) -> str:
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response


def parse_vlm_response(response: str) -> dict:
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        pass
    return None


# Draw bboxes on image
def draw_bboxes(image_path: str, annotations: list, temp_dir: str) -> str:
    img = cv2.imread(image_path)
    if img is None:
        return None

    for i, ann in enumerate(annotations):
        bbox = ann["bbox"]
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)

        label = f"#{ann['id']}"
        font_scale = max(0.8, min(img.shape[0], img.shape[1]) / 1000)
        thickness = max(2, int(font_scale * 2))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x, y - th - 10), (x + tw + 6, y), (0, 0, 255), -1)
        cv2.putText(img, label, (x + 3, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), thickness)

    temp_path = os.path.join(temp_dir, "phase3_temp.jpg")
    cv2.imwrite(temp_path, img)
    return temp_path


# Build prompt
def build_prompt(image_path: str, annotations: list, temp_dir: str) -> list:
    temp_path = draw_bboxes(image_path, annotations, temp_dir)
    if not temp_path:
        return None

    ann_text = "Annotation info for each bbox:\n"
    for ann in annotations:
        bbox = ann["bbox"]
        ann_text += f'  #{ann["id"]} text="{ann["text"]}" bbox=[{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]\n'

    prompt_text = f"""
This image has red bboxes with numbers drawn on it.
{ann_text}
Verify each annotation:
1. Does the actual text in the bbox match the annotation text?
2. Does the bbox accurately wrap the text? (not too loose, not cut off)
3. Is the text in the bbox clear and readable?

Also judge the overall image quality.

Answer in JSON format only:
{{
  "annotations": [
    {{"id": 1, "status": "correct", "reason": "text matches, bbox accurate"}},
    {{"id": 2, "status": "reject", "reason": "actual text is 커피, not 카페"}}
  ],
  "image_quality": "good"
}}

status: "correct" (accurate), "reject" (wrong), "pending" (hard to judge)
image_quality: "good", "blur", "dark", "occluded"
"""

    content = [
        {"type": "image", "image": f"file://{temp_path}"},
        {"type": "text", "text": prompt_text},
    ]
    messages = [{"role": "user", "content": content}]
    return messages


# Merge Phase 1b + Phase 2 results
def load_surviving_images(phase1b_paths: list, phase2_path: str) -> list:
    # 1. Phase 1b: collect images to remove (duplicates, keep only best)
    remove_images = set()
    for phase1b_path in phase1b_paths:
        if phase1b_path and os.path.exists(phase1b_path):
            with open(phase1b_path, "r", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("is_duplicate") is True:
                        best = r.get("best_image", -1)
                        images = r.get("images", [])
                        if best > 0 and best <= len(images):
                            for i, img in enumerate(images, 1):
                                if i != best:
                                    remove_images.add(img)
    print(f"  Phase 1b: {len(remove_images):,} images to remove ({len(phase1b_paths)} files)", flush=True)

    # 2. Phase 2: load records, filter out removed images and rejected annotations
    records = []
    total_images = 0
    total_keep_ann = 0
    skipped_images = 0

    with open(phase2_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            total_images += 1

            # Skip if removed by Phase 1b
            if r["image"] in remove_images:
                skipped_images += 1
                continue

            # Keep only "keep" annotations from Phase 2
            keep_anns = [a for a in r["annotations"] if a["status"] == "keep"]
            if not keep_anns:
                skipped_images += 1
                continue

            total_keep_ann += len(keep_anns)
            records.append({
                "image": r["image"],
                "label": r["label"],
                "width": r["width"],
                "height": r["height"],
                "annotations": keep_anns,
            })

    print(f"  Phase 2: {total_images:,} total images", flush=True)
    print(f"  Skipped: {skipped_images:,} (duplicates + zero keep)", flush=True)
    print(f"  Surviving: {len(records):,} images, {total_keep_ann:,} annotations", flush=True)
    return records


# Main
def main():
    parser = argparse.ArgumentParser(description="Phase 3: VLM annotation verification")
    parser.add_argument("--phase1b", type=str, nargs="*", default=[],
                        help="Phase 1b VLM dedup result JSONL(s), multiple files OK")
    parser.add_argument("--phase2", type=str, required=True,
                        help="Phase 2 annotation filter result JSONL")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path")
    parser.add_argument("--temp_dir", type=str, default="/scratch2/hklee2/vlm_temp",
                        help="Temp image directory")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index (for parallel runs)")
    parser.add_argument("--end_idx", type=int, default=0,
                        help="End index (0=all)")
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    os.makedirs("results", exist_ok=True)
    output_path = args.output or f"results/phase3_vlm_verify_{job_id}.jsonl"
    os.makedirs(args.temp_dir, exist_ok=True)

    print("\nVLM annotation verification...", flush=True)
    print(f"Phase 1b: {args.phase1b if args.phase1b else '(none)'}", flush=True)
    print(f"Phase 2:  {args.phase2}", flush=True)
    print(f"Model:    {args.model_name}", flush=True)

    # 1. Load surviving images
    print("\nLoading surviving images...", flush=True)
    records = load_surviving_images(args.phase1b, args.phase2)

    # Apply index range
    if args.end_idx > 0:
        records = records[args.start_idx:args.end_idx]
        print(f"  Range: [{args.start_idx}, {args.end_idx}) = {len(records):,} images", flush=True)

    # 2. Load model
    model, processor = load_model(args.model_name)

    # 3. Process
    total = len(records)
    print(f"\nProcessing: {total:,} images", flush=True)

    correct_count = 0
    reject_count = 0
    pending_count = 0
    quality_counts = {"good": 0, "blur": 0, "dark": 0, "occluded": 0, "unknown": 0}

    with open(output_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            image_path = record["image"]
            annotations = record["annotations"]

            try:
                messages = build_prompt(image_path, annotations, args.temp_dir)
                if messages is None:
                    continue

                response = call_vlm(model, processor, messages)
                parsed = parse_vlm_response(response)

                if parsed is None:
                    result_anns = []
                    for ann in annotations:
                        result_anns.append({
                            "id": ann["id"],
                            "bbox": ann["bbox"],
                            "text": ann["text"],
                            "status": "pending",
                            "reason": "vlm_parse_error",
                        })
                        pending_count += 1
                    img_quality = "unknown"
                else:
                    vlm_anns = {a["id"]: a for a in parsed.get("annotations", [])}
                    img_quality = parsed.get("image_quality", "unknown")

                    result_anns = []
                    for ann in annotations:
                        vlm_ann = vlm_anns.get(ann["id"], {})
                        status = vlm_ann.get("status", "pending")
                        reason = vlm_ann.get("reason", "")

                        if status not in ("correct", "reject", "pending"):
                            status = "pending"

                        result_anns.append({
                            "id": ann["id"],
                            "bbox": ann["bbox"],
                            "text": ann["text"],
                            "status": status,
                            "reason": reason,
                        })

                        if status == "correct":
                            correct_count += 1
                        elif status == "reject":
                            reject_count += 1
                        else:
                            pending_count += 1

            except Exception as e:
                result_anns = []
                for ann in annotations:
                    result_anns.append({
                        "id": ann["id"],
                        "bbox": ann["bbox"],
                        "text": ann["text"],
                        "status": "pending",
                        "reason": f"vlm_error: {str(e)[:100]}",
                    })
                    pending_count += 1
                img_quality = "unknown"

            if img_quality not in quality_counts:
                img_quality = "unknown"
            quality_counts[img_quality] += 1

            result = {
                "image": record["image"],
                "label": record["label"],
                "width": record["width"],
                "height": record["height"],
                "annotations": result_anns,
                "image_quality": img_quality,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()

            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1:,} / {total:,}", flush=True)

    # Statistics
    total_anns = correct_count + reject_count + pending_count
    print(f"\n{'-' * 60}", flush=True)
    print(f"Total images:      {total:,}", flush=True)
    print(f"Total annotations: {total_anns:,}", flush=True)
    print(f"  correct:         {correct_count:,}", flush=True)
    print(f"  reject:          {reject_count:,}", flush=True)
    print(f"  pending:         {pending_count:,}", flush=True)

    print(f"\nImage quality:", flush=True)
    for q, c in sorted(quality_counts.items(), key=lambda x: -x[1]):
        if c > 0:
            print(f"  {q:15s}: {c:,}", flush=True)

    print(f"\nOutput: {output_path}", flush=True)


if __name__ == "__main__":
    main()
