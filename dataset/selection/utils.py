import json
from pathlib import Path


def read_jsonl(path: str | Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def build_image_lookup(source_roots: list[Path]) -> dict[str, Path]:
    """filename -> full path 매핑 딕셔너리 생성."""
    lookup = {}
    for root in source_roots:
        for img in root.rglob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                lookup[img.name] = img
    return lookup
