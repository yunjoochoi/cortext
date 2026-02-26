"""Crop text regions from images and preprocess for PPOCR recognition."""

import cv2
import numpy as np
from pathlib import Path

from utils import read_jsonl

TARGET_HEIGHT = 48
TARGET_WIDTH = 320


def crop_and_preprocess(image_path: str, bbox: list[int]) -> np.ndarray:
    """Crop bbox from image, resize to PPOCR input format.

    Returns: np.ndarray of shape [3, 48, 320], float32, normalized.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read: {image_path}")

    x, y, w, h = bbox
    ih, iw = img.shape[:2]
    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(iw, int(x + w))
    y2 = min(ih, int(y + h))
    crop = img[y1:y2, x1:x2]

    if crop.size == 0:
        crop = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)

    crop = _resize_pad(crop, TARGET_HEIGHT, TARGET_WIDTH)
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - 0.5) / 0.5  # normalize to [-1, 1]
    crop = crop.transpose(2, 0, 1)  # HWC -> CHW
    return crop


def batch_iterator(manifest_path: str | Path, batch_size: int = 256):
    """Yield (records, preprocessed_crops) batches from manifest."""
    batch_records = []
    batch_crops = []

    for rec in read_jsonl(manifest_path):
        try:
            crop = crop_and_preprocess(rec["image_path"], rec["bbox"])
        except (FileNotFoundError, Exception):
            continue

        batch_records.append(rec)
        batch_crops.append(crop)

        if len(batch_records) == batch_size:
            yield batch_records, np.stack(batch_crops)
            batch_records, batch_crops = [], []

    if batch_records:
        yield batch_records, np.stack(batch_crops)


def _resize_pad(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize keeping aspect ratio, pad right side with zeros."""
    h, w = img.shape[:2]
    ratio = target_h / h
    new_w = min(int(w * ratio), target_w)
    resized = cv2.resize(img, (new_w, target_h))

    padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    padded[:, :new_w] = resized
    return padded
