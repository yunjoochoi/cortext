import json
import random
from pathlib import Path



TEXT_CONNECTORS = [
    ", content and position of the texts are ",
    ", textual material depicted in the image are ",
    ", texts that say ",
    ", captions shown in the snapshot are ",
    ", with the words of ",
    ", that reads ",
    ", the written materials on the picture: ",
    ", these texts are written on it: ",
    ", captions are ",
    ", content of the text in the graphic is ",
]


def build_prompt(caption: str, texts: list[str]) -> str:
    text_str = ", ".join(f"'{t}'" for t in texts)
    connector = random.choice(TEXT_CONNECTORS)
    if caption:
        return f"{caption}{connector}{text_str}"
    return f"A signage photo{connector}{text_str}"


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
    lookup = {}
    for root in source_roots:
        for img in root.rglob("*"):
            if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                lookup[img.name] = img
    return lookup
