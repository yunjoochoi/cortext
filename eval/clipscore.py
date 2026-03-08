# Adapted from https://github.com/jmhessel/clipscore/blob/1036465276513621f77f1c2208d742e4a430781f/clipscore.py
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
import clip
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import os
import warnings
from packaging import version
import ujson


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--img_dir",
        type=str,
        default='/data/vdb/yuxiang.tyx/AIGC/eval/gen_imgs_folder',
        help='path of generated images for eval'
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default='/data/vdb/yuxiang.tyx/AIGC/data/wukong_word/test1k.json',
        help='json path for evaluation dataset'
    )
    args = parser.parse_args()
    return args


def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf8') as f:
        content = ujson.load(f)
    return content


args = parse_args()
img_dir = args.img_dir
input_json = args.input_json
gpu = 'cuda:0'
num_samples = 4

model, transform = clip.load("ViT-B/32", device=gpu, jit=False)
# model, transform = clip.load("ViT-L/14", device=gpu, jit=False)
model.eval()


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if prefix != '' and self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions, prefix=''),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w*np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def cal_clipscore(image_ids, image_paths, text_list, device=None, references=None, scale_weight=2.5):

    image_feats = extract_all_images(image_paths, model, device, batch_size=64, num_workers=8)

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(model, image_feats, text_list, device, w=scale_weight)

    if references:
        pass

    else:
        scores = {image_id: {'CLIPScore': float(clipscore)}
                  for image_id, clipscore in
                  zip(image_ids, per_instance_image_text)}
        print('CLIPScore: {:.4f}'.format(np.mean([s['CLIPScore'] for s in scores.values()])))

    return scores


def eval_clipscore():
    clip_scores = []
    content = load_json(input_json)
    for i in tqdm.tqdm(range(num_samples)):
        img_ids = [gt['img_name'].split('.')[0]+f'_{i}.jpg' for gt in content['data_list']]
        captions = [gt['caption'] for gt in content['data_list']]
        img_paths = [os.path.join(img_dir, name) for name in img_ids]
        score = cal_clipscore(image_ids=img_ids, image_paths=img_paths, text_list=captions, device=gpu)
        clip_score = np.mean([s['CLIPScore'] for s in score.values()])
        clip_scores.append(clip_score)
    print(f"Mean clip_score: {np.mean(clip_scores):.4f}", )
    return np.mean(clip_scores)


if __name__ == '__main__':
    eval_clipscore()
