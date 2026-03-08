"""AnyText2 evaluation using PP-OCRv5(Korean)."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2
from cldm.recognizer import crop_image
from anytext2_singleGPU import load_data, get_item
from tqdm import tqdm
import torch
import Levenshtein
import numpy as np
import argparse
from paddleocr import PaddleOCR

PRINT_DEBUG = False
num_samples = 4


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        required=True,
        help='path of generated images for eval'
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help='json path for evaluation dataset'
    )
    parser.add_argument(
        "--lang",
        type=str,
        default='ch',
        help='language for PaddleOCR: ch, en, korean, etc.'
    )
    args = parser.parse_args()
    return args


def get_ld(s1, s2):
    """Normalized Edit Distance"""
    edit_dist = Levenshtein.distance(s1, s2)
    return 1 - edit_dist / (max(len(s1), len(s2)) + 1e-5)


def crop_text_region(img_tensor, pos_mask):
    """Crop text region from image using position mask."""
    np_pos = (pos_mask * 255.).astype(np.uint8)
    cropped = crop_image(img_tensor, np_pos)  # CHW tensor 0-255
    # convert to HWC numpy uint8
    cropped_np = cropped.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    # ensure 3 channels
    if cropped_np.shape[2] == 1:
        cropped_np = np.repeat(cropped_np, 3, axis=2)
    return cropped_np


def recognize_texts(ocr, crop_imgs):
    results = []
    for img in crop_imgs:
        # Use rec=True only (no detection needed since already cropped)
        rst = ocr.ocr(img, det=False, cls=True)
        if rst and rst[0]:
            # rst[0] is list of (text, confidence) tuples
            text = rst[0][0][0] if rst[0][0] else ''
        else:
            text = ''
        results.append(text)
    return results


def main():
    args = parse_args()
    img_dir = args.img_dir
    input_json = args.input_json

    ocr = PaddleOCR(
        ocr_version='PP-OCRv5',
        lang=args.lang,
        use_angle_cls=True,
        show_log=False,
    )
    print(f'PaddleOCR v5 initialized, lang={args.lang}')

    data_list = load_data(input_json)
    sen_acc = []
    edit_dist = []

    for i in tqdm(range(len(data_list)), desc='evaluate'):
        item_dict = get_item(data_list, i)
        img_name = item_dict['img_name'].split('.')[0]
        n_lines = item_dict['n_lines']
        gt_texts = []
        crop_imgs = []

        for j in range(num_samples):
            img_path = os.path.join(img_dir, img_name + f'_{j}.jpg')
            img = cv2.imread(img_path)
            if img is None:
                print(f'Warning: cannot read {img_path}, skipping...')
                continue
            if PRINT_DEBUG:
                cv2.imwrite(f'{i}_{j}.jpg', img)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()  # HWC -> CHW

            for k in range(n_lines):
                gt_texts.append(item_dict['texts'][k])
                cropped = crop_text_region(img_tensor, item_dict['positions'][k])
                crop_imgs.append(cropped)

        if n_lines > 0 and len(crop_imgs) > 0:
            preds_all = recognize_texts(ocr, crop_imgs)

            if PRINT_DEBUG:
                for idx, ci in enumerate(crop_imgs):
                    cv2.imwrite(f'{i}_{idx}_crop.jpg', ci)

            for k in range(len(preds_all)):
                pred_text = preds_all[k]
                gt_text = gt_texts[k]

                if pred_text == gt_text:
                    sen_acc.append(1)
                else:
                    sen_acc.append(0)

                ed = get_ld(pred_text, gt_text)
                edit_dist.append(ed)

                if PRINT_DEBUG:
                    print(f'pred/gt="{pred_text}"/"{gt_text}", ed={ed:.4f}')

    print(f'Done, lines={len(sen_acc)}, sen_acc={np.array(sen_acc).mean():.4f}, edit_dist={np.array(edit_dist).mean():.4f}')


if __name__ == "__main__":
    main()
