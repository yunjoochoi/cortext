"""Inference test for SDXL Korean text rendering LoRA checkpoint."""

import argparse
import os
import sys

import torch
from PIL import Image
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, set_peft_model_state_dict

# Reuse jamo definitions and decomposition from training script
sys.path.insert(0, os.path.dirname(__file__))
from train_korean_text_lora import (
    ALL_JAMO_TOKENS, EMPTY_TOKEN, CHOSEONG, JUNGSEONG, JONGSEONG,
    decompose_korean_char, decompose_text_to_jamo_sequences,
)
from models.jamo_combiner import JamoCombiner


BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
PROMPT_PREFIX = "A signage photo with '"
PROMPT_SUFFIX = "' written on it."


def import_model_class(pretrained, subfolder="text_encoder"):
    config = PretrainedConfig.from_pretrained(pretrained, subfolder=subfolder)
    name = config.architectures[0]
    if name == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif name == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection
        return CLIPTextModelWithProjection
    raise ValueError(f"Unsupported: {name}")


def load_checkpoint(checkpoint_dir, base_model=BASE_MODEL, device="cuda", dtype=torch.bfloat16):
    """Load base SDXL + LoRA checkpoint + jamo embeddings + combiners."""

    # Tokenizers (from checkpoint, 52 special tokens)
    tokenizer_one = AutoTokenizer.from_pretrained(
        os.path.join(checkpoint_dir, "tokenizer"), use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(
        os.path.join(checkpoint_dir, "tokenizer_2"), use_fast=False)

    # Text encoders
    te_cls_one = import_model_class(base_model, "text_encoder")
    te_cls_two = import_model_class(base_model, "text_encoder_2")
    text_encoder_one = te_cls_one.from_pretrained(base_model, subfolder="text_encoder")
    text_encoder_two = te_cls_two.from_pretrained(base_model, subfolder="text_encoder_2")

    # Resize embeddings to match tokenizer (+ 52 special tokens)
    text_encoder_one.resize_token_embeddings(len(tokenizer_one))
    text_encoder_two.resize_token_embeddings(len(tokenizer_two))

    # Load new token embeddings
    orig_vs_one = len(tokenizer_one) - 52
    orig_vs_two = len(tokenizer_two) - 52
    for name, te, orig_vs in [
        ("new_token_embeddings_one.pt", text_encoder_one, orig_vs_one),
        ("new_token_embeddings_two.pt", text_encoder_two, orig_vs_two),
    ]:
        path = os.path.join(checkpoint_dir, name)
        new_embeds = torch.load(path, map_location="cpu")
        te.text_model.embeddings.token_embedding.weight.data[orig_vs:] = new_embeds

    text_encoder_one.to(device, dtype=dtype).eval()
    text_encoder_two.to(device, dtype=dtype).eval()

    # UNet + LoRA
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    lora_config = LoraConfig(
        r=8, lora_alpha=8,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)

    from safetensors.torch import load_file
    lora_weights = load_file(os.path.join(checkpoint_dir, "pytorch_lora_weights.safetensors"))
    # Convert keys: diffusers format -> peft format
    peft_state = convert_unet_state_dict_to_peft(lora_weights)
    set_peft_model_state_dict(unet, peft_state, adapter_name="default")
    unet.to(device, dtype=dtype).eval()

    # JamoCombiners
    embed_dim_one = text_encoder_one.config.hidden_size
    embed_dim_two = text_encoder_two.config.hidden_size
    jamo_combiner_one = JamoCombiner(embed_dim=embed_dim_one, num_jamo_slots=3)
    jamo_combiner_two = JamoCombiner(embed_dim=embed_dim_two, num_jamo_slots=3)
    jamo_combiner_one.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "jamo_combiner_one.pt"), map_location="cpu"))
    jamo_combiner_two.load_state_dict(
        torch.load(os.path.join(checkpoint_dir, "jamo_combiner_two.pt"), map_location="cpu"))
    jamo_combiner_one.to(device, dtype=torch.float32).eval()
    jamo_combiner_two.to(device, dtype=torch.float32).eval()

    # VAE
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    vae.to(device, dtype=torch.float32).eval()

    # Noise scheduler
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # Jamo token id lookups
    jamo_ids_one = {tok: tokenizer_one.convert_tokens_to_ids(tok) for tok in ALL_JAMO_TOKENS}
    jamo_ids_two = {tok: tokenizer_two.convert_tokens_to_ids(tok) for tok in ALL_JAMO_TOKENS}

    return {
        "tokenizer_one": tokenizer_one,
        "tokenizer_two": tokenizer_two,
        "text_encoder_one": text_encoder_one,
        "text_encoder_two": text_encoder_two,
        "unet": unet,
        "vae": vae,
        "scheduler": scheduler,
        "jamo_combiner_one": jamo_combiner_one,
        "jamo_combiner_two": jamo_combiner_two,
        "jamo_ids_one": jamo_ids_one,
        "jamo_ids_two": jamo_ids_two,
    }


def encode_prompt_with_jamo(korean_texts, components, device, max_jamo_chars=20):
    """Encode prompt with jamo injection (same logic as training)."""
    PLACEHOLDER = EMPTY_TOKEN

    prompt_embeds_list = []
    pooled_prompt_embeds = None

    for enc_idx, (tokenizer, text_encoder, jamo_combiner, jamo_ids_map) in enumerate([
        (components["tokenizer_one"], components["text_encoder_one"],
         components["jamo_combiner_one"], components["jamo_ids_one"]),
        (components["tokenizer_two"], components["text_encoder_two"],
         components["jamo_combiner_two"], components["jamo_ids_two"]),
    ]):
        tok_emb = text_encoder.text_model.embeddings.token_embedding
        max_len = tokenizer.model_max_length
        placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER)

        n_fixed = len(tokenizer(
            f"{PROMPT_PREFIX}{PROMPT_SUFFIX}", add_special_tokens=True
        ).input_ids)
        max_chars = min(max_jamo_chars, max_len - n_fixed)

        # Build placeholder prompt from text list
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
        prompt_str = f"{PROMPT_PREFIX}{placeholder_str}{PROMPT_SUFFIX}"

        tok_out = tokenizer(
            [prompt_str], padding="max_length", max_length=max_len,
            truncation=True, return_tensors="pt",
        ).to(device)
        input_ids = tok_out.input_ids

        # Compute jamo combined embeddings for placeholder positions
        replacement_map = {}
        placeholder_mask = (input_ids[0] == placeholder_id)
        placeholder_positions = placeholder_mask.nonzero(as_tuple=True)[0]
        num_replacements = min(len(placeholder_positions), len(all_jamo_seqs))

        if num_replacements > 0:
            jamo_triple_ids = torch.tensor(
                [[jamo_ids_map[j] for j in all_jamo_seqs[i]]
                 for i in range(num_replacements)],
                dtype=torch.long, device=device,
            )
            jamo_embs = tok_emb(jamo_triple_ids)
            combined = jamo_combiner(jamo_embs.unsqueeze(0).float())
            combined = combined.squeeze(0)

            for i in range(num_replacements):
                replacement_map[(0, placeholder_positions[i].item())] = \
                    combined[i].to(tok_emb.weight.dtype)

        # Monkey-patch embeddings
        original_emb_forward = text_encoder.text_model.embeddings.forward

        def patched_emb_forward(input_ids=None, position_ids=None, inputs_embeds=None,
                                _replacement_map=replacement_map, _orig=original_emb_forward,
                                _tok_emb=tok_emb):
            if inputs_embeds is None:
                inputs_embeds = _tok_emb(input_ids)
            for (b, pos), emb in _replacement_map.items():
                inputs_embeds[b, pos] = emb
            return _orig(input_ids=input_ids, position_ids=position_ids,
                         inputs_embeds=inputs_embeds)

        text_encoder.text_model.embeddings.forward = patched_emb_forward
        try:
            with torch.no_grad():
                outputs = text_encoder(input_ids=input_ids, output_hidden_states=True)
        finally:
            text_encoder.text_model.embeddings.forward = original_emb_forward

        if enc_idx == 1:
            pooled_prompt_embeds = outputs.text_embeds
        hidden_states = outputs.hidden_states[-2]
        prompt_embeds_list.append(hidden_states)

    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)
    return prompt_embeds, pooled_prompt_embeds


@torch.no_grad()
def generate(
    korean_texts: list[str],
    components: dict,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    width: int = 1024,
    height: int = 768,
    seed: int | None = None,
    device: str = "cuda",
):
    """Generate image with Korean text rendering."""
    unet = components["unet"]
    vae = components["vae"]
    scheduler = components["scheduler"]

    # Encode prompt
    prompt_embeds, pooled_prompt_embeds = encode_prompt_with_jamo(
        korean_texts, components, device)

    # Encode negative prompt
    tokenizer_one = components["tokenizer_one"]
    tokenizer_two = components["tokenizer_two"]
    te_one = components["text_encoder_one"]
    te_two = components["text_encoder_two"]

    neg_embeds_list = []
    neg_pooled = None
    for enc_idx, (tok, te) in enumerate([(tokenizer_one, te_one), (tokenizer_two, te_two)]):
        neg_ids = tok(
            [""], padding="max_length", max_length=tok.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)
        outputs = te(input_ids=neg_ids, output_hidden_states=True)
        if enc_idx == 1:
            neg_pooled = outputs.text_embeds
        neg_embeds_list.append(outputs.hidden_states[-2])
    neg_prompt_embeds = torch.concat(neg_embeds_list, dim=-1)

    # SDXL time_ids
    add_time_ids = torch.tensor(
        [[height, width, 0, 0, height, width]],  # original_size + crop_top_left + target_size
        dtype=prompt_embeds.dtype, device=device,
    )

    # Scheduler setup
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # Latent initialization
    latent_h, latent_w = height // 8, width // 8
    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    latents = torch.randn(
        (1, unet.config.in_channels, latent_h, latent_w),
        generator=generator, device=device, dtype=prompt_embeds.dtype,
    )
    latents = latents * scheduler.init_noise_sigma

    # Denoising loop
    for t in timesteps:
        # CFG: concat unconditional + conditional
        latent_input = torch.cat([latents, latents])
        latent_input = scheduler.scale_model_input(latent_input, t)

        combined_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])
        combined_pooled = torch.cat([neg_pooled, pooled_prompt_embeds])
        combined_time_ids = torch.cat([add_time_ids, add_time_ids])

        noise_pred = unet(
            latent_input, t, combined_embeds,
            added_cond_kwargs={"time_ids": combined_time_ids, "text_embeds": combined_pooled},
            return_dict=False,
        )[0]

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Decode latents
    latents = latents / vae.config.scaling_factor
    image = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=BASE_MODEL)
    parser.add_argument("--text", type=str, required=True, help="Korean text (space-separated for multiple)")
    parser.add_argument("--output", type=str, default="/data/yunju/test_output.png")
    parser.add_argument("--num_images", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint from {args.checkpoint}...")
    components = load_checkpoint(args.checkpoint, args.base_model, device)

    korean_texts = args.text.split()
    print(f"Generating with texts: {korean_texts}")

    for i in range(args.num_images):
        seed = args.seed + i if args.seed is not None else None
        image = generate(
            korean_texts, components,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
            width=args.width, height=args.height,
            seed=seed, device=device,
        )
        if args.num_images == 1:
            out_path = args.output
        else:
            base, ext = os.path.splitext(args.output)
            out_path = f"{base}_{i}{ext}"
        image.save(out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
