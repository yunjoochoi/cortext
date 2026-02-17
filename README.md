# cortext

Dataset preprocessing and curation pipeline for Korean text rendering.

## Overview

`cortext` is a preprocessing and curation pipeline for building high-quality datasets for Korean text rendering models. It takes existing datasets and refines them into higher-quality data through advanced curation methods.

## Project Structure

```
cortext/
├── src/              # Core utility modules
│   └── utils.py
├── eval/             # Rendering quality evaluation scripts
│   └── z_image_test.py
├── docs/             # Documentation
├── script.sh         # SLURM cluster job submission script
└── pyproject.toml    # Project configuration
```

## Features

- **Data Preprocessing**: Format conversion and normalization of existing Korean text-image datasets
- **Data Curation**: Advanced quality-based filtering and refinement to produce higher-quality training data
- **Rendering Evaluation**: Quantitative quality assessment of text rendering outputs

## Setup

```bash
# Requires Python 3.13+
# Uses uv package manager
uv sync
```

## Usage

```bash
# HPC cluster (SLURM)
sbatch script.sh

# Local execution
python eval/z_image_test.py
```

## Requirements

- Python >= 3.13
- CUDA-enabled GPU (for evaluation scripts)
