"""Evaluate text accuracy of generated images using PP-OCRv5.

Crops text regions from generated images using GT bboxes,
runs OCR recognition, computes sentence accuracy and edit distance.
"""

import argparse
import json
from pathlib import Path

import cv2
import Levenshtein
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, required=True, help="Directory of generated images")
    p.add_argument("--eval_jsonl", type=str, required=True, help="Eval dataset jsonl")
    p.add_argument("--num_samples", type=int, default=4, help="Samples per image")
    p.add_argument("--lang", type=str, default="korean")
    p.add_argument("--output", type=str, default=None, help="Output results jsonl (optional)")
    return p.parse_args()


def normalized_edit_distance(s1: str, s2: str) -> float:
    d = Levenshtein.distance(s1, s2)
    return 1 - d / (max(len(s1), len(s2)) + 1e-5)


def crop_bbox(img: np.ndarray, bbox: list[int | float]) -> np.ndarray:
    """Crop COCO-format bbox [x, y, w, h] from image."""
    h_img, w_img = img.shape[:2]
    x, y, bw, bh = [int(v) for v in bbox]
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(w_img, x + bw)
    y1 = min(h_img, y + bh)
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return crop


def recognize_crop(ocr: PaddleOCR, crop: np.ndarray) -> str:
    result = ocr.ocr(crop, det=False, cls=True)
    if result and result[0]:
        return result[0][0][0] if result[0][0] else ""
    return ""


def main():
    args = parse_args()
    img_dir = Path(args.img_dir)

    records = []
    with open(args.eval_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} eval records")

    ocr = PaddleOCR(ocr_version="PP-OCRv5", lang=args.lang, use_angle_cls=True, show_log=False)
    print(f"PP-OCRv5 initialized, lang={args.lang}")

    sen_acc_all = []
    edit_dist_all = []
    results = []

    for rec in tqdm(records, desc="Evaluating"):
        stem = Path(rec["image_path"]).stem
        bbox_dict = rec.get("bbox", {})
        texts = rec.get("text", [])
        if isinstance(texts, str):
            texts = [texts]
        gt_w = rec.get("width", 1024)
        gt_h = rec.get("height", 1024)

        for s in range(args.num_samples):
            img_path = img_dir / f"{stem}_{s}.jpg"
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gen_h, gen_w = img.shape[:2]
            sx, sy = gen_w / gt_w, gen_h / gt_h

            for text_key, bbox in bbox_dict.items():
                scaled_bbox = [bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]
                crop = crop_bbox(img, scaled_bbox)
                if crop is None:
                    continue

                pred = recognize_crop(ocr, crop)
                gt = text_key
                exact = pred == gt
                ed = normalized_edit_distance(pred, gt)

                sen_acc_all.append(1 if exact else 0)
                edit_dist_all.append(ed)

                results.append({
                    "image": f"{stem}_{s}.jpg",
                    "gt": gt,
                    "pred": pred,
                    "exact": exact,
                    "edit_dist": ed,
                })

    sen_acc = np.mean(sen_acc_all) if sen_acc_all else 0.0
    edit_dist = np.mean(edit_dist_all) if edit_dist_all else 0.0
    print(f"\nResults: lines={len(sen_acc_all)}, sen_acc={sen_acc:.4f}, edit_dist={edit_dist:.4f}")

    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = {"sen_acc": sen_acc, "edit_dist": edit_dist, "n_lines": len(sen_acc_all)}
        summary_path = output_path.with_suffix(".summary.json")
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"Details -> {output_path}, Summary -> {summary_path}")


if __name__ == "__main__":
    main()
