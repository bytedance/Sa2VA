# Visual Reasoning Tracer (VRT)

[\[🏠 Project Page\]](https://harboryuan.github.io/visual-reasoning-tracer/)

## Overview

Current Multimodal Large Language Models (MLLMs) often lack transparency in their reasoning processes. To bridge this gap, we introduce **Visual Reasoning Tracer (VRT)**, a task requiring models to explicitly predict intermediate objects in a reasoning path. We present **VRT-Bench** for evaluation, a new metric for reasoning quality, and **VRT-80k**, a large-scale training dataset. Models trained on VRT-80k demonstrate significant improvements in grounded reasoning.

## Model Training

To train the model, use the following command:

```bash
bash tools/dist.sh projects/vrt_sa2va/configs_sa2va/vrt_sa2va_4b_qwen3_sft.py 8
```

## Directory Structure

- `configs_sa2va/`: Configuration files for Sa2VA models adapted for visual reasoning.
- `data_loader/`: Data loading scripts and utilities.
- `evaluation/`: Evaluation scripts and metrics for assessing reasoning capabilities.
- `models/`: Model definitions and architectures.
- `prompt/`: Prompt templates and engineering for visual reasoning tasks.
- `utils/`: Utility functions and helper scripts.

## Getting Started

Please refer to the [Project Page](https://harboryuan.github.io/visual-reasoning-tracer/) for more detailed information, including the paper, dataset, and additional resources.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{yuan2025vrt,
  author    = {Haobo Yuan and Yueyi Sun and Yanwei Li and Tao Zhang and Xueqing Deng and Henghui Ding and Lu Qi and Anran Wang and Xiangtai Li and Ming-Hsuan Yang},
  title     = {Visual Reasoning Tracer: Object-Level Grounded Reasoning Benchmark},
  journal   = {arXiv pre-print},
  year      = {2025},
}
```
