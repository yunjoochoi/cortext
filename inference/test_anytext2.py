'''Simple inference test for AnyText2'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import cv2
from ms_wrapper import AnyText2Model

print("=== AnyText2 Inference Test ===")
print("Loading model...")
inference = AnyText2Model(
    model_dir='./models',
    use_fp16=True,
    use_translator=False,
    font_path='font/Arial_Unicode.ttf',
    model_path='models/anytext_v2.0.ckpt'
).cuda(0)
print("Model loaded successfully!")

# Create a simple position image (white rectangle on black background)
pos_img = np.zeros((512, 512, 3), dtype=np.uint8)
cv2.rectangle(pos_img, (100, 200), (400, 300), (255, 255, 255), -1)

input_data = {
    "img_prompt": "A sign on a wall",
    "text_prompt": 'that reads "Hello"',
    "seed": 12345,
    "draw_pos": pos_img,
    "ori_image": None,
}

params = {
    "mode": "gen",
    "sort_priority": "↕",
    "show_debug": False,
    "revise_pos": False,
    "image_count": 1,
    "ddim_steps": 20,
    "image_width": 512,
    "image_height": 512,
    "strength": 1.0,
    "attnx_scale": 1.0,
    "font_hollow": True,
    "cfg_scale": 7.5,
    "eta": 0.0,
    "a_prompt": "best quality, extremely detailed, 4k, HD, supper legible text, clear text edges, clear strokes, neat writing, no watermarks",
    "n_prompt": "low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text",
    "glyline_font_path": [],
    "font_hint_image": [],
    "font_hint_mask": [],
    "text_colors": "",
}

print("Running inference...")
results, rtn_code, rtn_warning, debug_info = inference(input_data, **params)

if rtn_code >= 0:
    os.makedirs('test_output', exist_ok=True)
    for i, img in enumerate(results):
        out_path = f'test_output/result_{i}.png'
        cv2.imwrite(out_path, img[..., ::-1])
        print(f"Saved: {out_path}")
    print("=== Test PASSED ===")
else:
    print(f"=== Test FAILED: {rtn_warning} ===")
