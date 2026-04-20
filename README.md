# VRT-3D-VLM

Official implementation of **VRT-3D-VLM: Visual Reference Tokens with Explicit 3D Geometric Bias for Vision-Language Driving**.

## Overview

VRT-3D-VLM is a vision-language driving project built around a spatially grounded visual reference interface. In this release you will find:

- a paper-aligned `vrt3d/` package for model and trainer code,
- preprocessing utilities for dataset conversion,
- a supervised training entrypoint,
- token activation visualization tools for qualitative inspection.

## Installation

Python `3.10+` is recommended.

Create an environment and install the main dependencies:

```bash
conda create -n vrt3d python=3.10 -y
conda activate vrt3d

# Install PyTorch first according to your CUDA version.
# Example:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install \
  transformers datasets accelerate peft deepspeed \
  pillow opencv-python matplotlib requests tqdm \
  pycocotools pyquaternion shapely nuscenes-devkit
```

## Training

The main training entrypoint is:

```bash
python train_vrt3d.py \
  --model_name_or_path <BASE_MODEL_OR_CHECKPOINT> \
  --data_file_paths ./processed_datas/train.jsonl \
  --image_folders <IMAGE_ROOT> \
  --depth_info_dir ./processed_datas/depth_infos \
  --output_dir ./outputs/vrt3d \
  --attn_implementation flash_attention_2 \
  --learning_rate 1e-6 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 1
```

Notes:

- `data_file_paths` and `image_folders` support colon-separated lists if you want to combine multiple sources.
- Additional options are defined in [`vrt3d/trainer/vrt3d_sft_config.py`](vrt3d/trainer/vrt3d_sft_config.py).


## Acknowledgements

This release was developed with inspiration from prior open-source efforts in spatial vision-language modeling and decodable visual tokenization, especially:

- `n3d_vlm`
- `padt`
- Hugging Face `transformers`

Please also consider citing the original projects and papers that influenced this codebase.
