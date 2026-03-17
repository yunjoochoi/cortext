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


POS_LABELS = {
    0: " top left",
    1: " top",
    2: " top right",
    3: " left",
    4: None,
    5: " right",
    6: " bottom left",
    7: " bottom",
    8: " bottom right",
}


def build_prompt(caption: str, texts: list[str], pos_idxs: list[int] | None = None) -> str:
    base = caption if caption else "A signage photo"
    connector = random.choice(TEXT_CONNECTORS)

    if pos_idxs and len(pos_idxs) == len(texts):
        parts = []
        for text, pos in zip(texts, pos_idxs):
            if pos is None:
                parts.append(f"'{text}'")
                continue
            pos_label = POS_LABELS[pos]
            if pos_label is None:
                pos_label = random.choice([" middle", " center"])
            loc = random.choice([" located", " placed", " positioned", ""])
            prep = random.choice([" at", " in", " on"])
            parts.append(f"'{text}'{loc}{prep}{pos_label}")
            print("in build_prompt, in utils.py : ", base + connector + ", ".join(parts) + ".")
        
        return base + connector + ", ".join(parts) + "."

    text_str = ", ".join(f"'{t}'" for t in texts)
    return base + connector + text_str


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
