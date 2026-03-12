"""Test whether Qwen3 text encoder distinguishes similar Korean spellings."""

import torch
import torch.nn.functional as F
from transformers import Qwen2Tokenizer, Qwen3Model

MODEL_PATH = "/scratch2/shaush/models/models--Tongyi-MAI--Z-Image/snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021"
DEVICE = "cuda"
DTYPE = torch.bfloat16

PAIRS = [
    ("백조명품옷수선", "백존명품홋수선"),  # per-char substitution
    ("백조명품옷수선", "백조명품옷수선"),  # identical (upper bound)
    ("독수리약국", "독수니약국"),          # 1-char substitution
    ("안식처", "인식처"),                  # cho: ㅇ→ㅇ (actually ㅏ→ㅣ)
    ("열선 씨트", "열선 씨트"),            # identical
    ("커피숍", "터피숍"),                  # ㅋ→ㅌ
]


def main():
    print("Loading text encoder...")
    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    text_encoder = Qwen3Model.from_pretrained(MODEL_PATH, subfolder="text_encoder")
    text_encoder.requires_grad_(False).to(DEVICE, dtype=DTYPE)

    print(f"\n{'Text A':>20} | {'Text B':>20} | {'cos_sim':>8} | {'l2_dist':>8} | tokens_A | tokens_B")
    print("-" * 100)

    for text_a, text_b in PAIRS:
        tok_a = tokenizer(text_a, return_tensors="pt", padding=True).to(DEVICE)
        tok_b = tokenizer(text_b, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            emb_a = text_encoder(**tok_a).last_hidden_state.mean(dim=1).float()
            emb_b = text_encoder(**tok_b).last_hidden_state.mean(dim=1).float()

        cos = F.cosine_similarity(emb_a, emb_b).item()
        l2 = (emb_a - emb_b).norm().item()

        n_tok_a = tok_a["input_ids"].shape[1]
        n_tok_b = tok_b["input_ids"].shape[1]

        print(f"{text_a:>20} | {text_b:>20} | {cos:>8.4f} | {l2:>8.2f} | {n_tok_a:>8} | {n_tok_b:>8}")

    # Token-level comparison for first pair
    print("\n--- Token-level analysis for first pair ---")
    for text in [PAIRS[0][0], PAIRS[0][1]]:
        ids = tokenizer.encode(text)
        tokens = [tokenizer.decode([i]) for i in ids]
        print(f"  '{text}' -> {len(ids)} tokens: {tokens}")

    print("\nDone.")


if __name__ == "__main__":
    main()
