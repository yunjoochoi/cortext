"""
Phase 1b: VLM-based duplicate verification.

Input: phase1a hash groups JSONL
Output: VLM duplicate verification results JSONL

Usage:
    python phase1b_vlm_dedup.py \
        --hash_groups results/phase1a_hash_groups_XXXX.jsonl \
        --model_name Qwen/Qwen3-VL-8B-Instruct
"""

import argparse
import json
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


# VLM model loading
def load_model(model_name: str):
    print(f"\nLoading model...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"  Model loaded: {model_name}", flush=True)
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
        output_ids = model.generate(**inputs, max_new_tokens=256)

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
    return {"is_duplicate": None, "best_image": -1, "reason": f"parse_error: {response[:200]}"}


# Prompt building
def build_prompt(image_paths: list) -> list:
    content = []

    for idx, img_path in enumerate(image_paths, 1):
        content.append({"type": "image", "image": f"file://{img_path}"})
        content.append({"type": "text", "text": f"Image {idx}: {os.path.basename(img_path)}"})

    prompt_text = """
These images were flagged as potentially duplicates by hash comparison.

Step 1: Are these images of the same scene/subject?
Step 2: If yes, select the best image based on:
  1. Text clarity and readability
  2. Text not occluded or cut off
  3. Image sharpness and brightness

Answer in JSON format only:
- Same scene: {"is_duplicate": true, "best_image": <image number>, "reason": "why"}
- Different scene: {"is_duplicate": false, "best_image": -1, "reason": "why"}
"""
    content.append({"type": "text", "text": prompt_text})
    messages = [{"role": "user", "content": content}]
    return messages


# Main
def main():
    parser = argparse.ArgumentParser(description="Phase 1b: VLM duplicate verification")
    parser.add_argument("--hash_groups", type=str, required=True,
                        help="Phase 1a hash groups JSONL")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path")
    parser.add_argument("--hash_types", nargs="+",
                        default=["md5", "phash_exact", "phash_similar", "dhash_exact", "dhash_similar"],
                        help="Hash types to process")
    parser.add_argument("--max_group_size", type=int, default=10,
                        help="Skip groups larger than this (default: 10)")
    parser.add_argument("--n_samples", type=int, default=0,
                        help="Max groups per hash type (0=all)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start index for parallel runs")
    parser.add_argument("--end_idx", type=int, default=0,
                        help="End index (0=all)")
    args = parser.parse_args()

    job_id = os.environ.get("SLURM_JOB_ID", "local")
    os.makedirs("results", exist_ok=True)
    output_path = args.output or f"results/phase1b_vlm_dedup_{job_id}.jsonl"

    print("\nVLM duplicate verification...", flush=True)
    print(f"Input:          {args.hash_groups}", flush=True)
    print(f"Model:          {args.model_name}", flush=True)
    print(f"Max group size: {args.max_group_size}", flush=True)

    # 1. Load hash groups
    print("\nLoading hash groups...", flush=True)
    groups_by_type = {}
    with open(args.hash_groups, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            ht = record["hash_type"]
            if ht not in groups_by_type:
                groups_by_type[ht] = []
            groups_by_type[ht].append(record)

    # Flatten all groups into single list for index-based splitting
    all_groups = []
    for ht in args.hash_types:
        if ht in groups_by_type:
            for g in groups_by_type[ht]:
                all_groups.append(g)

    print(f"  Total groups: {len(all_groups):,}", flush=True)

    # Apply index range
    if args.end_idx > 0:
        all_groups = all_groups[args.start_idx:args.end_idx]
        print(f"  Range: [{args.start_idx}, {args.end_idx}) = {len(all_groups):,} groups", flush=True)

    # Re-group by hash type
    groups_by_type = {}
    for g in all_groups:
        ht = g["hash_type"]
        if ht not in groups_by_type:
            groups_by_type[ht] = []
        groups_by_type[ht].append(g)

    for ht, groups in groups_by_type.items():
        print(f"  {ht}: {len(groups)} groups", flush=True)

    # 2. Load model
    model, processor = load_model(args.model_name)

    # 3. Flatten all processable groups
    all_process_groups = []
    total_skipped = 0
    for ht in args.hash_types:
        if ht not in groups_by_type:
            continue
        groups = groups_by_type[ht]
        if args.n_samples > 0:
            groups = groups[:args.n_samples]
        for group in groups:
            if group["count"] > args.max_group_size:
                total_skipped += 1
            else:
                all_process_groups.append(group)

    total_to_process = len(all_process_groups)
    print(f"\nProcessing: {total_to_process:,} groups (skipped: {total_skipped:,})", flush=True)

    # 4. Process
    total_processed = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for i, group in enumerate(all_process_groups):
            images = group["images"]
            hash_type = group["hash_type"]

            try:
                messages = build_prompt(images)
                response = call_vlm(model, processor, messages)
                parsed = parse_vlm_response(response)
            except Exception as e:
                parsed = {"is_duplicate": None, "best_image": -1, "reason": f"vlm_error: {str(e)[:200]}"}

            result = {
                "hash_type": hash_type,
                "group_id": group["group_id"],
                "count": len(images),
                "images": images,
                "is_duplicate": parsed.get("is_duplicate"),
                "best_image": parsed.get("best_image", -1),
                "reason": parsed.get("reason", ""),
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            total_processed += 1

            if (i + 1) % 500 == 0:
                print(f"  Progress: {i + 1:,} / {total_to_process:,}", flush=True)

    # Statistics
    print(f"\n{'-' * 60}", flush=True)
    print(f"Processed: {total_processed:,}", flush=True)
    print(f"Skipped:   {total_skipped:,}", flush=True)
    print(f"Output:    {output_path}", flush=True)


if __name__ == "__main__":
    main()
