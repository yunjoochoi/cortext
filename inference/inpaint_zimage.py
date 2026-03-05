"""Z-Image bbox inpainting: replace text region with new styled text."""

import argparse
import inspect
from pathlib import Path

import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    ZImagePipeline,
    ZImageTransformer2DModel,
)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def retrieve_timesteps(scheduler, num_inference_steps, device, **kwargs):
    scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, num_inference_steps


def build_latent_mask(
    bbox: list[int],
    orig_w: int,
    orig_h: int,
    latent_h: int,
    latent_w: int,
) -> torch.Tensor:
    """Pixel bbox [x, y, w, h] -> binary mask [1, 1, latent_H, latent_W]."""
    x, y, w, h = bbox
    sx = latent_w / orig_w
    sy = latent_h / orig_h
    lx1 = max(0, round(x * sx))
    ly1 = max(0, round(y * sy))
    lx2 = min(latent_w, round((x + w) * sx))
    ly2 = min(latent_h, round((y + h) * sy))
    mask = torch.zeros(1, 1, latent_h, latent_w)
    if lx2 > lx1 and ly2 > ly1:
        mask[0, 0, ly1:ly2, lx1:lx2] = 1.0
    return mask


def encode_image(vae: AutoencoderKL, image: Image.Image, height: int, width: int, device, dtype) -> torch.Tensor:
    """Resize image and encode to latent space."""
    image = image.convert("RGB").resize((width, height), Image.LANCZOS)
    from torchvision.transforms import functional as TF
    img_tensor = TF.to_tensor(image).unsqueeze(0).to(device=device, dtype=dtype)
    img_tensor = img_tensor * 2.0 - 1.0  # [0,1] -> [-1,1]
    latents = vae.encode(img_tensor).latent_dist.sample()
    latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
    return latents


def inpaint(
    pipe: ZImagePipeline,
    image: Image.Image,
    bbox: list[int],
    prompt: str,
    negative_prompt: str = "",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    strength: float = 1.0,
    mask_blur: int = 4,
    seed: int = 42,
) -> Image.Image:
    """Inpaint bbox region of image using Z-Image with given prompt.

    Args:
        strength: 0.0=keep original, 1.0=full denoise. Controls how much noise to add.
        mask_blur: gaussian blur kernel size for mask feathering (0=no blur).
    """
    device = pipe._execution_device
    dtype = pipe.transformer.dtype
    generator = torch.Generator(device=device).manual_seed(seed)

    # Encode source image
    orig_w, orig_h = image.size
    init_latents = encode_image(pipe.vae, image, height, width, device, torch.float32)

    # Build latent mask
    vae_scale = pipe.vae_scale_factor * 2  # Z-Image uses *2
    latent_h = height // vae_scale
    latent_w = width // vae_scale
    # latent shape from VAE is [B, C, H, W] where H = height/vae_scale, W = width/vae_scale
    # But Z-Image VAE has block_out_channels that define a scale factor of 8, and image_processor uses *2
    # Actual latent spatial dims:
    latent_h_actual = init_latents.shape[2]
    latent_w_actual = init_latents.shape[3]

    mask = build_latent_mask(bbox, orig_w, orig_h, latent_h_actual, latent_w_actual)
    mask = mask.to(device=device, dtype=torch.float32)

    # Optional mask blur for smoother blending
    if mask_blur > 0:
        import torch.nn.functional as F
        k = mask_blur * 2 + 1
        mask = F.avg_pool2d(mask, kernel_size=k, stride=1, padding=k // 2)
        mask = (mask > 0.01).float()  # re-binarize after blur, or keep soft for smoother edges

    # Encode prompt
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt,
        negative_prompt=negative_prompt if guidance_scale > 0 else None,
        do_classifier_free_guidance=guidance_scale > 0,
        device=device,
    )

    # Prepare timesteps
    image_seq_len = (latent_h_actual // 2) * (latent_w_actual // 2)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.sigma_min = 0.0
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler, num_inference_steps, device, mu=mu,
    )

    # Apply strength: skip early timesteps (less noise = more preservation)
    if strength < 1.0:
        start_step = int(num_inference_steps * (1.0 - strength))
        timesteps = timesteps[start_step:]
        num_inference_steps = len(timesteps)

    # Add noise to init_latents at the first timestep
    noise = torch.randn_like(init_latents, generator=generator)
    t_start = timesteps[0]
    # Flow matching: x_t = (1 - t/1000) * x_0 + (t/1000) * noise
    sigma = t_start / 1000.0
    latents = (1.0 - sigma) * init_latents + sigma * noise

    # Denoising loop with mask blending
    for i, t in enumerate(timesteps):
        timestep = t.expand(latents.shape[0])
        timestep_norm = (1000 - timestep) / 1000

        apply_cfg = guidance_scale > 0

        if apply_cfg:
            latent_input = latents.to(dtype).repeat(2, 1, 1, 1)
            prompt_input = prompt_embeds + negative_prompt_embeds
            t_input = timestep_norm.repeat(2)
        else:
            latent_input = latents.to(dtype)
            prompt_input = prompt_embeds
            t_input = timestep_norm

        latent_input = latent_input.unsqueeze(2)
        latent_input_list = list(latent_input.unbind(dim=0))

        model_out = pipe.transformer(
            latent_input_list, t_input, prompt_input, return_dict=False
        )[0]

        if apply_cfg:
            pos_out = model_out[:1]
            neg_out = model_out[1:]
            noise_pred = torch.stack([p.float() + guidance_scale * (p.float() - n.float())
                                      for p, n in zip(pos_out, neg_out)])
        else:
            noise_pred = torch.stack([o.float() for o in model_out])

        noise_pred = noise_pred.squeeze(2)
        noise_pred = -noise_pred

        # Scheduler step -> predicted x_{t-1}
        latents_denoised = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Blend: masked region = denoised, unmasked = noised original
        if i < len(timesteps) - 1:
            t_next = timesteps[i + 1]
            sigma_next = t_next / 1000.0
            noised_orig = (1.0 - sigma_next) * init_latents + sigma_next * noise
            latents = mask * latents_denoised + (1.0 - mask) * noised_orig
        else:
            # Last step: final blend with clean original
            latents = mask * latents_denoised + (1.0 - mask) * init_latents

    # Decode
    latents = latents.to(pipe.vae.dtype)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    decoded = pipe.vae.decode(latents, return_dict=False)[0]
    result = pipe.image_processor.postprocess(decoded, output_type="pil")[0]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Z-Image model path")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA weights path")
    parser.add_argument("--image", type=str, required=True, help="Source image path")
    parser.add_argument("--bbox", type=int, nargs=4, required=True, help="x y w h in pixel coords")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for inpainted region")
    parser.add_argument("--negative_prompt", type=str, default="")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--strength", type=float, default=0.8, help="0=keep original, 1=full denoise")
    parser.add_argument("--mask_blur", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="inpainted.png")
    args = parser.parse_args()

    pipe = ZImagePipeline.from_pretrained(args.model_path, torch_dtype=torch.bfloat16)
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)
    pipe.to("cuda")

    image = Image.open(args.image)
    result = inpaint(
        pipe=pipe,
        image=image,
        bbox=args.bbox,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        mask_blur=args.mask_blur,
        seed=args.seed,
    )
    result.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
