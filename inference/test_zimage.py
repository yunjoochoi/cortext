"""Simple Z-Image text-to-image inference."""

import torch
from diffusers import ZImagePipeline


pipe = ZImagePipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.enable_model_cpu_offload()

prompt = (
    'A photorealistic medium shot of a dusty, leather-bound old hardcover book. '
    'Aligned vertically along the center of the book\'s narrow spine, there is very small, '
    'elegant, thin serif English text that reads: "Rsdfjweoi of the sfersergs". '
    'The gold-leaf lettering of the text is slightly faded and worn by time, yet still legible '
    'against the dark leather texture. Subtle ambient lighting, high detail, 8k resolution.'
)

generator = torch.Generator(device="cpu").manual_seed(42)
image = pipe(prompt=prompt, generator=generator, height=1024, width=768).images[0]
image.save("zimage_book_test.png")
print("Saved zimage_book_test.png")
