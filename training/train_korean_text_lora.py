#!/home/yunju/.conda/envs/zimage/bin/python
"""Korean text rendering LoRA fine-tuning with jamo-level special tokens (SDXL).

52 special tokens (51 Korean jamo + 1 empty token) are added to BOTH CLIP
tokenizers. A learnable JamoCombiner (Linear) maps 3 jamo embeddings
(초성, 중성, 종성) into 1 character embedding per syllable.

Backbone: Stable Diffusion XL (dual text encoder, 1024px default).
Trainable parameters: UNet LoRA + new token embeddings (both encoders) + JamoCombiner.

Usage:
    accelerate launch training/train_korean_text_lora.py \
        --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
        --manifest /data/yunju/manifest.jsonl \
        --output_dir /data/yunju/ckpt \
        --train_batch_size 2 --gradient_accumulation_steps 8 \
        --max_train_steps 10000 --learning_rate 1e-4 \
        --rank 8 --mixed_precision bf16 \
        --gradient_checkpointing --checkpointing_steps 1000

manifest.jsonl format (one per line, one record per image):
    {"image_path": "/path/to/img.jpg", "text": ["커피숖", "샛별"], "bbox": {"커피숖": [x,y,w,h], ...}, "width": 1600, "height": 1200, ...}
"""

import argparse
import json
import logging
import math
import os
import random
import shutil
import unicodedata
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.loaders import StableDiffusionLoraLoaderMixin
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers, convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module

from models.jamo_combiner import JamoCombiner

logger = get_logger(__name__)


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


# ---------------------------------------------------------------------------
# Korean Jamo definitions 
# ---------------------------------------------------------------------------
# 14 basic consonants
JAMO_CONSONANTS = list("ㄱㄴㄷㄹㅁㅂㅅㅇㅈㅊㅋㅌㅍㅎ")
# 10 basic vowels
JAMO_VOWELS = list("ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣ")
# 5 double consonants
JAMO_DOUBLE_CONSONANTS = list("ㄲㄸㅃㅆㅉ")
# 11 compound vowels
JAMO_COMPOUND_VOWELS = list("ㅐㅒㅔㅖㅘㅙㅚㅝㅞㅟㅢ")
# 11 compound final consonants
JAMO_COMPOUND_FINALS = list("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")

EMPTY_TOKEN = "<EMPTY>"

ALL_JAMO_TOKENS = (
    JAMO_CONSONANTS + JAMO_VOWELS + JAMO_DOUBLE_CONSONANTS
    + JAMO_COMPOUND_VOWELS + JAMO_COMPOUND_FINALS + [EMPTY_TOKEN]
)
assert len(ALL_JAMO_TOKENS) == 52, f"Expected 52 tokens, got {len(ALL_JAMO_TOKENS)}"

# ---------------------------------------------------------------------------
# Hangul syllable decomposition tables
# ---------------------------------------------------------------------------
# 초성
CHOSEONG = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
]
# 중성
JUNGSEONG = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ',
]
# 종성
JONGSEONG = [
    None, 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
]


def decompose_korean_char(ch: str):
    """Decompose a single Korean syllable into (초성, 중성, 종성) jamo tokens.

    Returns a list of 3 jamo token strings. Non-decomposable characters
    (standalone jamo, non-Korean) return None.
    """
    code = ord(ch)
    # Hangul Syllables block
    if 0xAC00 <= code <= 0xD7A3:
        offset = code - 0xAC00
        cho_idx = offset // (21 * 28)
        jung_idx = (offset % (21 * 28)) // 28
        jong_idx = offset % 28
        cho = CHOSEONG[cho_idx]
        jung = JUNGSEONG[jung_idx]
        jong = JONGSEONG[jong_idx] if jong_idx != 0 else EMPTY_TOKEN
        return [cho, jung, jong]
    return None


def decompose_text_to_jamo_sequences(text: str):
    """Decompose Korean text into a list of (cho, jung, jong) triples.

    Non-Korean characters (spaces, punctuation, etc.) are skipped.
    Returns list of 3-element lists of jamo token strings.
    """
    result = []
    for ch in text:
        decomposed = decompose_korean_char(ch)
        if decomposed is not None:
            result.append(decomposed)
    return result


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Korean text rendering LoRA training")
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--variant", type=str, default=None)
    p.add_argument("--manifest", type=str, required=True,
                   help="Path to manifest.jsonl with image_path, text, optional caption")
    p.add_argument("--output_dir", type=str, default="korean-text-lora")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--center_crop", action="store_true")
    p.add_argument("--random_flip", action="store_true")
    p.add_argument("--train_batch_size", type=int, default=4)
    p.add_argument("--num_train_epochs", type=int, default=100)
    p.add_argument("--max_train_steps", type=int, default=None)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="cosine")
    p.add_argument("--lr_warmup_steps", type=int, default=100)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank")
    p.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (default: same as rank)")
    p.add_argument("--snr_gamma", type=float, default=None)
    p.add_argument("--noise_offset", type=float, default=0.0)
    p.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--checkpointing_steps", type=int, default=500)
    p.add_argument("--checkpoints_total_limit", type=int, default=10)
    p.add_argument("--resume_from_checkpoint", type=str, default=None)
    p.add_argument("--allow_tf32", action="store_true")
    p.add_argument("--report_to", type=str, default="tensorboard")
    p.add_argument("--logging_dir", type=str, default="logs")
    p.add_argument("--validation_prompt", type=str, default=None,
                   help="Korean text for validation, e.g. '안녕하세요'")
    p.add_argument("--validation_epochs", type=int, default=50)
    p.add_argument("--num_validation_images", type=int, default=4)
    p.add_argument("--use_8bit_adam", action="store_true")
    p.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    p.add_argument("--max_jamo_chars", type=int, default=20,
                   help="Maximum number of Korean characters (syllables) per sample")
    args = p.parse_args()
    if args.lora_alpha is None:
        args.lora_alpha = args.rank
    return args


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def build_prompt(texts: list[str], caption: str = "") -> str:
    joined = " ".join(texts)
    if caption:
        return f"{caption}, with '{joined}' written on it."
    return f"A signage photo with '{joined}' written on it."


# bucket: 1024 × 768
BUCKET_W, BUCKET_H = 1024, 768


class KoreanTextDataset(Dataset):
    def __init__(self, manifest_path: str, resolution: int,
                 center_crop: bool = False, random_flip: bool = False):
        self.records = []
        with open(manifest_path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("text"):
                    self.records.append(rec)
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(rec["image_path"])
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_size = (image.height, image.width)

        # Resize to fixed 1024×768 bucket
        image = transforms.functional.resize(
            image, (BUCKET_H, BUCKET_W),
            interpolation=transforms.InterpolationMode.BILINEAR,
        )

        if self.random_flip and random.random() < 0.5:
            image = transforms.functional.hflip(image)

        y1, x1 = 0, 0
        image = crop(image, y1, x1, BUCKET_H, BUCKET_W)
        crop_top_left = (y1, x1)

        texts = rec["text"]  # list of strings
        caption = rec.get("caption", "")
        return {
            "pixel_values": self.to_tensor(image),
            "text": texts,
            "prompt": build_prompt(texts, caption),
            "original_size": original_size,
            "crop_top_left": crop_top_left,
            "target_size": (BUCKET_H, BUCKET_W),
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    logging_dir = Path(args.output_dir, args.logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_dir)),
        kwargs_handlers=[kwargs],
    )
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)

    if args.seed is not None:
        set_seed(args.seed)
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # ---- Load SDXL models (dual text encoders) ----
    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer",
        revision=args.revision, use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2",
        revision=args.revision, use_fast=False)

    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2")

    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        revision=args.revision, variant=args.variant)
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2",
        revision=args.revision, variant=args.variant)

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        revision=args.revision, variant=args.variant)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet",
        revision=args.revision, variant=args.variant)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler")

    # ---- Add 52 Korean jamo special tokens to BOTH tokenizers & resize embeddings table ----
    num_added_one = tokenizer_one.add_tokens(ALL_JAMO_TOKENS)
    num_added_two = tokenizer_two.add_tokens(ALL_JAMO_TOKENS)
    text_encoder_one.resize_token_embeddings(len(tokenizer_one))
    text_encoder_two.resize_token_embeddings(len(tokenizer_two))
    logger.info(f"Added {num_added_one}/{num_added_two} jamo tokens to tokenizer_one/two")

    original_vocab_size_one = len(tokenizer_one) - num_added_one
    original_vocab_size_two = len(tokenizer_two) - num_added_two

    # Build jamo token id lookups (per tokenizer)
    jamo_token_ids_one = {tok: tokenizer_one.convert_tokens_to_ids(tok) for tok in ALL_JAMO_TOKENS}
    jamo_token_ids_two = {tok: tokenizer_two.convert_tokens_to_ids(tok) for tok in ALL_JAMO_TOKENS}

    # ---- JamoCombiner (one per text encoder, different hidden sizes) ----
    embed_dim_one = text_encoder_one.config.hidden_size   # 768 for CLIP ViT-L
    embed_dim_two = text_encoder_two.config.hidden_size   # 1280 for CLIP ViT-bigG
    jamo_combiner_one = JamoCombiner(embed_dim=embed_dim_one, num_jamo_slots=3)
    jamo_combiner_two = JamoCombiner(embed_dim=embed_dim_two, num_jamo_slots=3)

    # ---- Freeze base models ----
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)

    # Unfreeze new token embeddings only
    text_encoder_one.text_model.embeddings.token_embedding.weight.requires_grad = True
    text_encoder_two.text_model.embeddings.token_embedding.weight.requires_grad = True

    # SDXL VAE must stay in float32 to avoid NaN losses
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    jamo_combiner_one.to(accelerator.device, dtype=torch.float32)
    jamo_combiner_two.to(accelerator.device, dtype=torch.float32)

    if args.enable_xformers_memory_efficient_attention:
        from diffusers.utils.import_utils import is_xformers_available
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()

    # ---- LoRA on UNet ----
    lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        # Text encoders: skip gradient checkpointing — they are mostly frozen
        # (only new token embeddings are trainable) and manual forward breaks
        # with GradientCheckpointingLayer wrapping.
    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    # ---- Collect trainable parameters ----
    lora_params = list(filter(lambda p: p.requires_grad, unet.parameters()))
    te_one_embedding = text_encoder_one.text_model.embeddings.token_embedding
    te_two_embedding = text_encoder_two.text_model.embeddings.token_embedding
    combiner_params = list(jamo_combiner_one.parameters()) + list(jamo_combiner_two.parameters())

    all_trainable_params = [
        {"params": lora_params, "lr": args.learning_rate},
        {"params": [te_one_embedding.weight], "lr": args.learning_rate},
        {"params": [te_two_embedding.weight], "lr": args.learning_rate},
        {"params": combiner_params, "lr": args.learning_rate},
    ]

    # Gradient mask hooks so only new token embeddings get updated
    def make_mask_hook(orig_vocab_size):
        def hook(grad):
            mask = torch.zeros_like(grad)
            mask[orig_vocab_size:] = 1.0
            return grad * mask
        return hook

    te_one_embedding.weight.register_hook(make_mask_hook(original_vocab_size_one))
    te_two_embedding.weight.register_hook(make_mask_hook(original_vocab_size_two))

    # ---- Dataset & DataLoader ----
    train_dataset = KoreanTextDataset(
        args.manifest, args.resolution, args.center_crop, args.random_flip)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.dataloader_num_workers, drop_last=True,
    )
    logger.info(f"Dataset: {len(train_dataset):,} samples")

    # ---- Optimizer ----
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer = bnb.optim.AdamW8bit(all_trainable_params, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(all_trainable_params, weight_decay=1e-4)

    # ---- LR scheduler ----
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # ---- Accelerate prepare ----
    (unet, text_encoder_one, text_encoder_two, jamo_combiner_one, jamo_combiner_two,
     optimizer, train_dataloader, lr_scheduler) = accelerator.prepare(
        unet, text_encoder_one, text_encoder_two, jamo_combiner_one, jamo_combiner_two,
        optimizer, train_dataloader, lr_scheduler)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    if accelerator.is_main_process:
        accelerator.init_trackers("korean-text-lora-sdxl", config=vars(args))
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    # ---- Helpers ----
    def unwrap(model):
        m = accelerator.unwrap_model(model)
        return m._orig_mod if is_compiled_module(m) else m

    def compute_time_ids(original_size, crop_top_left, target_size):
        add_time_ids = list(original_size + crop_top_left + target_size)
        return torch.tensor([add_time_ids], dtype=weight_dtype, device=accelerator.device)

    # Pre-tokenize fixed English prefix/suffix (same for all samples)
    PROMPT_PREFIX = "A signage photo with '"
    PROMPT_SUFFIX = "' written on it."
    # Placeholder token — one <EMPTY> per Korean syllable in the prompt
    PLACEHOLDER = EMPTY_TOKEN  # "<EMPTY>"

    def _build_placeholder_prompt(korean_texts: list[str], max_chars: int) -> tuple:
        """Build prompt with <EMPTY> placeholders for Korean chars.

        Each text is decomposed into jamo, placeholders are joined by space.
        Returns (prompt_str, jamo_sequences).
        """
        all_jamo_seqs = []
        placeholder_parts = []
        remaining = max_chars
        for text in korean_texts:
            jamo_seqs = decompose_text_to_jamo_sequences(text)
            jamo_seqs = jamo_seqs[:remaining]
            if not jamo_seqs:
                continue
            all_jamo_seqs.extend(jamo_seqs)
            placeholder_parts.append(PLACEHOLDER * len(jamo_seqs))
            remaining -= len(jamo_seqs)
            if remaining <= 0:
                break
        placeholder_str = " ".join(placeholder_parts)
        prompt = f"{PROMPT_PREFIX}{placeholder_str}{PROMPT_SUFFIX}"
        return prompt, all_jamo_seqs

    def encode_prompt_with_jamo(prompt_texts, korean_texts, max_jamo_chars):
        """SDXL dual-encoder prompt encoding with placeholder token replacement.

        AnyText2-style approach:
          1. Build prompt with <EMPTY> placeholders for Korean chars
          2. Tokenize normally → input_ids preserved
          3. token_embedding(input_ids) → get standard embeddings
          4. Replace <EMPTY> positions with JamoCombiner(3 jamo embeddings)
          5. Monkey-patch embeddings layer to inject modified embeddings
          6. Call text_encoder(input_ids) normally (standard CLIP forward)

        Returns (prompt_embeds [B, 77, 2048], pooled_prompt_embeds [B, 1280]).
        """
        prompt_embeds_list = []
        pooled_prompt_embeds = None
        batch_size = len(korean_texts)

        for enc_idx, (tokenizer, text_encoder, jamo_combiner, jamo_ids_map) in enumerate([
            (tokenizer_one, text_encoder_one, jamo_combiner_one, jamo_token_ids_one),
            (tokenizer_two, text_encoder_two, jamo_combiner_two, jamo_token_ids_two),
        ]):
            te = unwrap(text_encoder)
            tok_emb = te.text_model.embeddings.token_embedding
            max_len = tokenizer.model_max_length  # 77
            placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER)

            n_fixed = len(tokenizer(
                f"{PROMPT_PREFIX}{PROMPT_SUFFIX}", add_special_tokens=True
            ).input_ids)  # BOS + prefix + suffix + EOS
            max_chars = min(max_jamo_chars, max_len - n_fixed)

            # Build prompts and tokenize
            all_jamo_seqs = []
            prompt_strs = []
            for b_idx in range(batch_size):
                prompt_str, jamo_seqs = _build_placeholder_prompt(
                    korean_texts[b_idx], max_chars)
                prompt_strs.append(prompt_str)
                all_jamo_seqs.append(jamo_seqs)

            tok_out = tokenizer(
                prompt_strs, padding="max_length", max_length=max_len,
                truncation=True, return_tensors="pt",
            ).to(accelerator.device)
            input_ids = tok_out.input_ids  # [B, 77]

            # Compute replacement embeddings for placeholder positions
            # For each sample: find <EMPTY> positions, compute JamoCombiner(jamo embs)
            replacement_map = {}  # (batch_idx, position) → combined_embedding
            for b_idx in range(batch_size):
                placeholder_mask = (input_ids[b_idx] == placeholder_id)
                placeholder_positions = placeholder_mask.nonzero(as_tuple=True)[0]
                jamo_seqs = all_jamo_seqs[b_idx]
                num_replacements = min(len(placeholder_positions), len(jamo_seqs))

                if num_replacements > 0:
                    # Build jamo triple ids: [num_replacements, 3]
                    jamo_triple_ids = torch.tensor(
                        [[jamo_ids_map[j] for j in jamo_seqs[i]]
                         for i in range(num_replacements)],
                        dtype=torch.long, device=accelerator.device,
                    )
                    # Lookup jamo embeddings: [num_replacements, 3, D]
                    jamo_embs = tok_emb(jamo_triple_ids)
                    # Combine 3 → 1: [1, num_replacements, 3, D] → [1, num_replacements, D]
                    combined = jamo_combiner(jamo_embs.unsqueeze(0).float())
                    combined = combined.squeeze(0)  # [num_replacements, D]

                    for i in range(num_replacements):
                        replacement_map[(b_idx, placeholder_positions[i].item())] = \
                            combined[i].to(tok_emb.weight.dtype)

            # Monkey-patch the embeddings layer to inject combined jamo embeddings
            # (same pattern as AnyText2's embedding_forward)
            original_emb_forward = te.text_model.embeddings.forward

            def patched_emb_forward(input_ids=None, position_ids=None, inputs_embeds=None,
                                    _replacement_map=replacement_map, _orig=original_emb_forward,
                                    _tok_emb=tok_emb):
                # Get standard token embeddings
                if inputs_embeds is None:
                    inputs_embeds = _tok_emb(input_ids)
                # Replace placeholder positions with JamoCombiner outputs
                for (b, pos), emb in _replacement_map.items():
                    inputs_embeds[b, pos] = emb
                # Continue with original forward (position embeddings etc.)
                # We call the original but pass our modified inputs_embeds
                return _orig(input_ids=input_ids, position_ids=position_ids,
                             inputs_embeds=inputs_embeds)
            te.text_model.embeddings.forward = patched_emb_forward

            try:
                # Standard CLIP forward — input_ids is intact, embeddings are patched
                outputs = te(
                    input_ids=input_ids,
                    output_hidden_states=True,
                )
            finally:
                # Restore original forward
                te.text_model.embeddings.forward = original_emb_forward

            if enc_idx == 1:
                # CLIPTextModelWithProjection: pooled output
                pooled_prompt_embeds = outputs.text_embeds
            # Penultimate hidden state (second-to-last encoder layer output)
            hidden_states = outputs.hidden_states[-2]
            prompt_embeds_list.append(hidden_states)

        # Concat along hidden dim: [B, 77, 768] + [B, 77, 1280] → [B, 77, 2048]
        prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
        return prompt_embeds, pooled_prompt_embeds

    # ---- Save/Load hooks (trainable params only) ----
    def save_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            # Save UNet LoRA weights
            unet_lora_state = get_peft_model_state_dict(unwrap(unet))
            StableDiffusionXLPipeline.save_lora_weights(
                output_dir, unet_lora_layers=unet_lora_state, safe_serialization=True)

            # Save new token embeddings from both encoders
            for name, te, orig_vs in [
                ("new_token_embeddings_one.pt", text_encoder_one, original_vocab_size_one),
                ("new_token_embeddings_two.pt", text_encoder_two, original_vocab_size_two),
            ]:
                te_unwrapped = unwrap(te)
                full_embeds = te_unwrapped.text_model.embeddings.token_embedding.weight.data
                torch.save(full_embeds[orig_vs:].cpu(), os.path.join(output_dir, name))

            # Save JamoCombiner (both)
            torch.save(unwrap(jamo_combiner_one).state_dict(),
                       os.path.join(output_dir, "jamo_combiner_one.pt"))
            torch.save(unwrap(jamo_combiner_two).state_dict(),
                       os.path.join(output_dir, "jamo_combiner_two.pt"))

            # Save tokenizers
            tokenizer_one.save_pretrained(os.path.join(output_dir, "tokenizer"))
            tokenizer_two.save_pretrained(os.path.join(output_dir, "tokenizer_2"))

            for _ in models:
                weights.pop()

    def load_hook(models, input_dir):
        # Load UNet LoRA
        lora_state_dict, _ = StableDiffusionLoraLoaderMixin.lora_state_dict(input_dir)
        unet_state = {k.replace("unet.", ""): v for k, v in lora_state_dict.items() if k.startswith("unet.")}
        unet_state = convert_unet_state_dict_to_peft(unet_state)

        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap(unet))):
                set_peft_model_state_dict(model, unet_state, adapter_name="default")

        # Load new token embeddings
        for name, te, orig_vs in [
            ("new_token_embeddings_one.pt", text_encoder_one, original_vocab_size_one),
            ("new_token_embeddings_two.pt", text_encoder_two, original_vocab_size_two),
        ]:
            path = os.path.join(input_dir, name)
            if os.path.exists(path):
                new_embeds = torch.load(path, map_location="cpu")
                te_unwrapped = unwrap(te)
                te_unwrapped.text_model.embeddings.token_embedding.weight.data[orig_vs:] = \
                    new_embeds.to(te_unwrapped.device)

        # Load JamoCombiner
        for fname, combiner in [
            ("jamo_combiner_one.pt", jamo_combiner_one),
            ("jamo_combiner_two.pt", jamo_combiner_two),
        ]:
            path = os.path.join(input_dir, fname)
            if os.path.exists(path):
                unwrap(combiner).load_state_dict(torch.load(path, map_location="cpu"))

        if args.mixed_precision == "fp16":
            cast_training_params([unwrap(unet)], dtype=torch.float32)

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    # ---- Resume ----
    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        path = args.resume_from_checkpoint
        if path == "latest":
            dirs = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None
        if path:
            accelerator.print(f"Resuming from {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(range(args.max_train_steps), initial=global_step, desc="Steps",
                        disable=not accelerator.is_local_main_process)

    # ---- Training loop ----
    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        jamo_combiner_one.train()
        jamo_combiner_two.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet, jamo_combiner_one, jamo_combiner_two):
                # Encode images → latents (VAE is float32)
                with torch.no_grad():
                    latents = vae.encode(
                        batch["pixel_values"].to(dtype=vae.dtype)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(dtype=weight_dtype)

                # Sample noise
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    noise += args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,),
                                          device=latents.device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # SDXL time ids
                add_time_ids = torch.cat([
                    compute_time_ids(s, c, t)
                    for s, c, t in zip(batch["original_sizes"], batch["crop_top_lefts"],
                                       batch["target_sizes"])
                ])

                # Dual-encoder prompt encoding with jamo injection
                prompt_embeds, pooled_prompt_embeds = encode_prompt_with_jamo(
                    batch["prompts"], batch["texts"], args.max_jamo_chars)

                unet_added_conditions = {
                    "time_ids": add_time_ids,
                    "text_embeds": pooled_prompt_embeds,
                }

                # Predict noise
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

                model_pred = unet(
                    noisy_latents, timesteps, prompt_embeds,
                    added_cond_kwargs=unet_added_conditions, return_dict=False,
                )[0]

                # Loss
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]
                    if noise_scheduler.config.prediction_type == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    all_clip_params = (
                        lora_params + [te_one_embedding.weight, te_two_embedding.weight]
                        + combiner_params
                    )
                    accelerator.clip_grad_norm_(all_clip_params, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        ckpts = sorted(
                            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")],
                            key=lambda x: int(x.split("-")[1]))
                        while len(ckpts) >= args.checkpoints_total_limit:
                            shutil.rmtree(os.path.join(args.output_dir, ckpts.pop(0)))
                    accelerator.save_state(os.path.join(args.output_dir, f"checkpoint-{global_step}"))
                    logger.info(f"Saved checkpoint-{global_step}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # ---- Validation ----
        if (accelerator.is_main_process and args.validation_prompt
                and epoch % args.validation_epochs == 0):
            logger.info(f"Running validation with text: {args.validation_prompt}")
            pipeline = StableDiffusionXLPipeline.from_pretrained(
                args.pretrained_model_name_or_path, torch_dtype=weight_dtype,
                revision=args.revision, variant=args.variant)
            pipeline.unet = unwrap(unet)
            pipeline.text_encoder = unwrap(text_encoder_one)
            pipeline.text_encoder_2 = unwrap(text_encoder_two)
            pipeline.tokenizer = tokenizer_one
            pipeline.tokenizer_2 = tokenizer_two
            pipeline = pipeline.to(accelerator.device)
            pipeline.set_progress_bar_config(disable=True)
            gen = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
            val_prompt = build_prompt([args.validation_prompt])
            images = [pipeline(val_prompt, generator=gen, num_inference_steps=30).images[0]
                      for _ in range(args.num_validation_images)]
            for tracker in accelerator.trackers:
                if tracker.name == "tensorboard":
                    import numpy as np
                    tracker.writer.add_images("validation",
                        np.stack([np.asarray(img) for img in images]), epoch, dataformats="NHWC")
            del pipeline
            torch.cuda.empty_cache()

    # ---- Final save ----
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save LoRA
        unet_lora_state = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrap(unet)))
        StableDiffusionXLPipeline.save_lora_weights(
            args.output_dir, unet_lora_layers=unet_lora_state, safe_serialization=True)

        # Save new token embeddings (both encoders)
        for name, te, orig_vs in [
            ("new_token_embeddings_one.pt", text_encoder_one, original_vocab_size_one),
            ("new_token_embeddings_two.pt", text_encoder_two, original_vocab_size_two),
        ]:
            te_unwrapped = unwrap(te)
            full_embeds = te_unwrapped.text_model.embeddings.token_embedding.weight.data
            torch.save(full_embeds[orig_vs:].cpu(), os.path.join(args.output_dir, name))

        # Save JamoCombiner (both)
        torch.save(unwrap(jamo_combiner_one).state_dict(),
                   os.path.join(args.output_dir, "jamo_combiner_one.pt"))
        torch.save(unwrap(jamo_combiner_two).state_dict(),
                   os.path.join(args.output_dir, "jamo_combiner_two.pt"))

        # Save tokenizers
        tokenizer_one.save_pretrained(os.path.join(args.output_dir, "tokenizer"))
        tokenizer_two.save_pretrained(os.path.join(args.output_dir, "tokenizer_2"))

        logger.info(f"All trainable weights saved to {args.output_dir}")

    accelerator.end_training()


def collate_fn(examples):
    # Bucket-aware batching: pad to largest in batch (all should be same bucket ideally)
    pixel_values = torch.stack([e["pixel_values"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {
        "pixel_values": pixel_values,
        "prompts": [e["prompt"] for e in examples],
        "texts": [e["text"] for e in examples],
        "original_sizes": [e["original_size"] for e in examples],
        "crop_top_lefts": [e["crop_top_left"] for e in examples],
        "target_sizes": [e["target_size"] for e in examples],
    }


if __name__ == "__main__":
    main()
