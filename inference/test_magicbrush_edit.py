"""Test MagicBrush (InstructPix2Pix) for bbox text editing vs Z-Image inpainting."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionInstructPix2PixPipeline


def create_bbox_highlight(image: Image.Image, bbox: list[int], color="red") -> Image.Image:
    """Draw bbox on image for visualization."""
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    x, y, w, h = bbox
    draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
    return vis


def edit_with_magicbrush(
    image: Image.Image,
    instruction: str,
    model_id: str = "timbrooks/instruct-pix2pix",
    num_inference_steps: int = 50,
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
    seed: int = 42,
) -> Image.Image:
    """Edit image using InstructPix2Pix / MagicBrush."""
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16, safety_checker=None,
    )
    pipe.to("cuda")

    generator = torch.Generator("cuda").manual_seed(seed)
    result = pipe(
        prompt=instruction,
        image=image,
        num_inference_steps=num_inference_steps,
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    return result


def main():
    parser = argparse.ArgumentParser(description="Test MagicBrush text editing")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--bbox", type=int, nargs=4, required=True, help="x y w h")
    parser.add_argument("--target_text", type=str, default="카페라떼")
    parser.add_argument("--model_id", type=str, default="timbrooks/instruct-pix2pix",
                        help="timbrooks/instruct-pix2pix or OSU-NLP-Group/MagicBrush")
    parser.add_argument("--image_guidance_scale", type=float, default=1.5,
                        help="How much to follow original image (higher=more faithful)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="How much to follow instruction (higher=stronger edit)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="edit_results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    bbox = args.bbox

    # Save bbox visualization
    bbox_vis = create_bbox_highlight(image, bbox)
    bbox_vis.save(output_dir / "00_bbox.png")

    # Sweep: different instructions for the same goal
    instructions = [
        f"Change the text on the sign to '{args.target_text}'",
        f"Replace the text with '{args.target_text}'",
        f"Edit the sign to say '{args.target_text}'",
        f"Make the text read '{args.target_text}'",
    ]

    # Sweep image_guidance_scale: controls original image preservation
    img_scales = [1.0, 1.5, 2.0]

    print(f"Target text: {args.target_text}")
    print(f"Model: {args.model_id}")
    print(f"Running {len(instructions)} instructions x {len(img_scales)} image scales = {len(instructions) * len(img_scales)} experiments")

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, safety_checker=None,
    )
    pipe.to("cuda")
    generator = torch.Generator("cuda")

    for i, instruction in enumerate(instructions):
        for j, igs in enumerate(img_scales):
            generator.manual_seed(args.seed)
            result = pipe(
                prompt=instruction,
                image=image,
                num_inference_steps=args.steps,
                image_guidance_scale=igs,
                guidance_scale=args.guidance_scale,
                generator=generator,
            ).images[0]

            fname = f"magicbrush_i{i}_igs{igs:.1f}.png"
            result.save(output_dir / fname)
            print(f"  [{fname}] instruction='{instruction}', image_guidance={igs}")

    print(f"\nResults saved to {output_dir}/")
    print("\nNOTE: InstructPix2Pix/MagicBrush edits the entire image, not just the bbox.")
    print("For bbox-only editing, use inpaint_zimage.py instead.")


if __name__ == "__main__":
    main()
