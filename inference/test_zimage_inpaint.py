import torch
import numpy as np
from PIL import Image
from diffusers import ZImageInpaintPipeline
from diffusers.utils import load_image

pipe = ZImageInpaintPipeline.from_pretrained("Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16)
pipe.to("cuda")

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
init_image = load_image(url).resize((1024, 1024))

# Create a mask (white = inpaint, black = preserve)
mask = np.zeros((1024, 1024), dtype=np.uint8)
mask[256:768, 256:768] = 255  # Inpaint center region
mask_image = Image.fromarray(mask)

prompt = "A beautiful lake with mountains in the background"
image = pipe(
    prompt,
    image=init_image,
    mask_image=mask_image,
    strength=1.0,
    num_inference_steps=9,
    guidance_scale=0.0,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
image.save("zimage_inpaint.png")
