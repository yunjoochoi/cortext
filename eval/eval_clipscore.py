# Adapted from https://github.com/jmhessel/clipscore
'''
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
'''

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import clip
import numpy as np
import sklearn.preprocessing
import torch
import tqdm
from packaging import version
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from core.utils import build_prompt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, required=True, help="Directory of generated images")
    p.add_argument("--eval_jsonl", type=str, required=True, help="Eval dataset jsonl")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--clip_model", type=str, default="ViT-B/32")
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix=""):
        self.data = data
        self.prefix = prefix
        if prefix and self.prefix[-1] != " ":
            self.prefix += " "

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {"caption": c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

    def __getitem__(self, idx):
        image = Image.open(self.data[idx])
        image = self.preprocess(image)
        return {"image": image}

    def __len__(self):
        return len(self.data)


def extract_text_features(captions, model, device, batch_size=256, num_workers=8):
    loader = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, prefix=""),
        batch_size=batch_size, num_workers=num_workers, shuffle=False,
    )
    all_feats = []
    with torch.no_grad():
        for b in tqdm.tqdm(loader, desc="Text features"):
            b = b["caption"].to(device)
            all_feats.append(model.encode_text(b).cpu().numpy())
    return np.vstack(all_feats)


def extract_image_features(image_paths, model, device, batch_size=64, num_workers=8):
    loader = torch.utils.data.DataLoader(
        CLIPImageDataset(image_paths),
        batch_size=batch_size, num_workers=num_workers, shuffle=False,
    )
    all_feats = []
    with torch.no_grad():
        for b in tqdm.tqdm(loader, desc="Image features"):
            b = b["image"].to(device)
            if device.startswith("cuda"):
                b = b.to(torch.float16)
            all_feats.append(model.encode_image(b).cpu().numpy())
    return np.vstack(all_feats)


def compute_clipscore(image_feats, text_feats, w=2.5):
    if version.parse(np.__version__) < version.parse("1.21"):
        image_feats = sklearn.preprocessing.normalize(image_feats, axis=1)
        text_feats = sklearn.preprocessing.normalize(text_feats, axis=1)
    else:
        image_feats = image_feats / np.sqrt(np.sum(image_feats**2, axis=1, keepdims=True))
        text_feats = text_feats / np.sqrt(np.sum(text_feats**2, axis=1, keepdims=True))
    per = w * np.clip(np.sum(image_feats * text_feats, axis=1), 0, None)
    return float(np.mean(per)), per


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

    model, _ = clip.load(args.clip_model, device=args.device, jit=False)
    model.eval()

    scores_per_sample = []
    for s in range(args.num_samples):
        image_paths = []
        captions = []
        for rec in records:
            stem = Path(rec["image_path"]).stem
            img_path = img_dir / f"{stem}_{s}.jpg"
            if img_path.exists():
                image_paths.append(str(img_path))
                caption = rec.get("caption", "")
                texts = rec.get("text", [])
                if isinstance(texts, str):
                    texts = [texts]
                captions.append(build_prompt(caption, texts))

        if not image_paths:
            print(f"  Sample {s}: no images found, skipping")
            continue

        img_feats = extract_image_features(image_paths, model, args.device)
        txt_feats = extract_text_features(captions, model, args.device)
        mean_score, _ = compute_clipscore(img_feats, txt_feats)
        scores_per_sample.append(mean_score)
        print(f"  Sample {s}: CLIPScore={mean_score:.4f} ({len(image_paths)} images)")

    overall = np.mean(scores_per_sample) if scores_per_sample else 0.0
    print(f"\nMean CLIPScore: {overall:.4f}")


if __name__ == "__main__":
    main()
