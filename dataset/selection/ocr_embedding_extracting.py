"""Phase 1: Extract PPOCR hidden state embeddings from text crops.

Builds PP-OCRv5 server rec model from vendored ppocr components,
loads pretrained weights, and hooks into the SVTR neck to capture
120-dim embeddings.
"""

import json
import numpy as np
import paddle
import yaml
from pathlib import Path
from tqdm import tqdm

from crop_and_preprocess import batch_iterator
from ppocr import build_model

SHARD_SIZE = 50_000
PRETRAINED_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/"
    "official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams"
)
_PPOCR_DIR = Path(__file__).parent / "ppocr"


class OCREmbeddingExtractor:
    def __init__(self, pretrained: str = PRETRAINED_URL, batch_size: int = 256):
        self.batch_size = batch_size
        self.neck_output = None
        self._load_model(pretrained)

    def _load_model(self, pretrained: str):
        pretrained = pretrained or PRETRAINED_URL
        config_path = _PPOCR_DIR / "PP-OCRv5_server_rec.yml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        arch_config = config["Architecture"]

        # char dict vendored alongside the config
        char_dict_path = _PPOCR_DIR / "ppocrv5_dict.txt"
        with open(char_dict_path) as f:
            n_chars = len(f.readlines())
        # +1 space (use_space_char=True), +1 CTC blank
        char_num = n_chars + 2

        arch_config["Head"]["out_channels_list"] = {
            "CTCLabelDecode": char_num,
            "SARLabelDecode": char_num + 2,
            "NRTRLabelDecode": char_num + 3,
        }

        model = build_model(arch_config)
        _load_pretrained(model, pretrained)
        model.eval()

        # Hook SVTR neck: MultiHead -> ctc_encoder -> encoder (EncoderWithSVTR)
        svtr_neck = model.head.ctc_encoder.encoder
        svtr_neck.register_forward_post_hook(self._capture_neck)
        self.rec_model = model

    def _capture_neck(self, layer, input, output):
        self.neck_output = output

    def extract_batch(self, crops: np.ndarray) -> np.ndarray:
        """Extract embeddings from preprocessed crops [B, 3, 48, 320].

        Returns [B, embedding_dim] float32 numpy array.
        """
        tensor = paddle.to_tensor(crops)
        with paddle.no_grad():
            self.rec_model(tensor)

        features = self.neck_output
        if features.ndim == 4:
            embeddings = paddle.mean(features, axis=[2, 3])
        elif features.ndim == 3:
            embeddings = paddle.mean(features, axis=1)
        else:
            embeddings = features
        return embeddings.numpy()

    def run(self, manifest_path: str, output_dir: str):
        """Extract embeddings for all annotations in manifest."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_ids = []
        all_embeddings = []
        shard_idx = 0

        for records, crops in tqdm(
            batch_iterator(manifest_path, self.batch_size),
            desc="Extracting embeddings",
        ):
            embs = self.extract_batch(crops)
            all_embeddings.append(embs)
            all_ids.extend(r["annotation_id"] for r in records)

            if len(all_ids) >= SHARD_SIZE:
                _save_shard(output_dir, shard_idx, all_ids, all_embeddings)
                shard_idx += 1
                all_ids, all_embeddings = [], []

        if all_ids:
            _save_shard(output_dir, shard_idx, all_ids, all_embeddings)

        _merge_shards(output_dir)
        print(f"Embeddings saved to {output_dir}")


def _load_pretrained(model, url_or_path: str):
    """Download if URL, then load .pdparams weights into model."""
    path = url_or_path
    if path.startswith("http"):
        # paddle handles caching in WEIGHT_HOME (~/.cache/paddle/weights/)
        print(f"Downloading pretrained weights (cached after first run)...")
        path = paddle.utils.download.get_weights_path_from_url(path)

    if path.endswith(".pdparams"):
        path = path[:-len(".pdparams")]
    params = paddle.load(path + ".pdparams")

    state_dict = model.state_dict()
    loaded = {}
    for k, v in params.items():
        if k not in state_dict:
            continue
        if list(v.shape) != list(state_dict[k].shape):
            continue
        loaded[k] = v.astype(state_dict[k].dtype) if v.dtype != state_dict[k].dtype else v

    model.set_state_dict(loaded)
    print(f"Loaded {len(loaded)}/{len(state_dict)} params")


def _save_shard(output_dir: Path, idx: int, ids: list, embeddings: list):
    np.save(output_dir / f"shard_{idx}.npy", np.concatenate(embeddings))
    with open(output_dir / f"shard_{idx}_ids.json", "w") as f:
        json.dump(ids, f)


def _merge_shards(output_dir: Path):
    shard_files = sorted(output_dir.glob("shard_*.npy"))
    id_files = sorted(output_dir.glob("shard_*_ids.json"))

    all_emb = np.concatenate([np.load(f) for f in shard_files])
    all_ids = []
    for f in id_files:
        with open(f) as fh:
            all_ids.extend(json.load(fh))

    np.save(output_dir / "embeddings.npy", all_emb)
    with open(output_dir / "embedding_ids.json", "w") as f:
        json.dump(all_ids, f)

    for f in shard_files:
        f.unlink()
    for f in id_files:
        f.unlink()
