# cortext

Dataset preprocessing and curation pipeline for Korean text rendering.

## Overview

`cortext` takes raw Korean text-image datasets and produces a curated coreset via k-Center Greedy selection in PP-OCRv5 embedding space — filtering out redundant/common glyphs to maximize visual diversity.

## Project Structure

```
cortext/
├── dataset/
│   └── selection/            # Coreset selection pipeline
│       ├── run_pipeline.py   # Orchestrator (Phase 1–3)
│       ├── utils.py          # JSONL I/O helpers
│       ├── prepare/
│       │   └── build_manifest.py   # Phase 0: manifest build
│       └── module/
│           ├── data_conditioning/  # Crop + OCR embedding (PP-OCRv5)
│           ├── coreset_selection/  # k-Center Greedy
│           └── output_finalization/
├── model/                    # Model training scripts
├── eval/                     # Evaluation scripts
├── configs/
│   └── selection_config.yaml
└── scripts/                  # SLURM job scripts
    ├── 1_install_paddle.sh
    ├── 2_run_manifest.sh
    └── 3_extract_embeds.sh
```

## Requirements

- Python 3.12
- CUDA 13.0 (Driver), CUDA 12.8 (Runtime)
- GPU with Compute Capability ≥ 8.0 (tested on RTX 3090)

### Python packages

| Package | Version |
|---|---|
| paddlepaddle-gpu | 3.3.0 |
| paddleocr | 3.4.0 |
| paddlex | 3.4.2 |
| torch | 2.7.1 |
| numpy | 2.4.2 |
| opencv-contrib-python | 4.10.0.84 |
| pillow | 12.1.1 |
| tqdm | 4.67.3 |

## Setup

### 1. Create conda environment

```bash
conda create -n cortext python=3.12
conda activate cortext
```

### 2. Install PaddlePaddle GPU

PaddlePaddle must be installed from the official index (not PyPI).
Match the CUDA version to your driver:

```bash
# CUDA 13.0 driver
pip install paddlepaddle-gpu==3.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/

# Verify
python -c "import paddle; print(paddle.__version__)"
```

### 3. Install remaining packages

```bash
pip install -r requirements.txt
```

Pretrained weights are downloaded automatically on first run (~200MB, cached at `~/.cache/paddle/`).

## Usage

Edit `configs/selection_config.yaml` to set data paths and `k`:

```yaml
data:
  data_root: "/path/to/dataset"
  output_dir: "/path/to/output"
selection:
  k: 100000
```

### Phase 0: Build manifest

```bash
python dataset/selection/prepare/build_manifest.py
# or: sbatch scripts/2_run_manifest.sh
```

Outputs `<output_dir>/manifest.jsonl` — one line per text bbox annotation.

### Phase 1–3: Extract embeddings → select coreset → export

```bash
python dataset/selection/run_pipeline.py
# or: sbatch scripts/3_extract_embeds.sh
```

Outputs:
- `<output_dir>/embeddings/embeddings.npy` — PP-OCRv5 SVTR embeddings (120-dim)
- `<output_dir>/selected_indices.json` — k-Center Greedy selected indices
- `<output_dir>/coreset_selected.jsonl` — final curated annotations
