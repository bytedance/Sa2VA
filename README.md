# Pixel LLMs: Pixel-Level Grounded Understanding for Multimodal LLMs

**Pixel LLMs** is a family of projects that bring pixel-level, dense grounded understanding to multimodal LLMs. It is anchored by **Sa2VA** — a unified model that marries SAM-2 with LLaVA for dense grounded understanding of images and videos — together with a growing set of research projects built on top of it.

![Teaser](assets/images/teaser.jpg)

## Projects

### 🧠 [Sa2VA](./projects/sa2va/README.md) — Marrying SAM2 with LLaVA (T-PAMI 2026)
*Haobo Yuan, Xiangtai Li, Tao Zhang, Yueyi Sun, Zilong Huang, Shilin Xu, Shunping Ji, Yunhai Tong, Lu Qi, Jiashi Feng, Ming-Hsuan Yang*

The core unified model: SAM-2 + MLLM for referring segmentation, grounded conversation, visual prompting, and image/video chat. Supports InternVL2.5/3 and Qwen2.5-VL/Qwen3-VL backbones.

📂 [`projects/sa2va`](./projects/sa2va/README.md) · [📜 arXiv](https://arxiv.org/abs/2501.04001) · [🏠 Page](https://lxtgh.github.io/project/sa2va) · [🤗 Models](https://huggingface.co/collections/ByteDance/sa2va-model-zoo-677e3084d71b5f108d00e093)

### 🔍 [VRT](./projects/vrt_sa2va/README.md) — Visual Reasoning Tracer
*Haobo Yuan, Yueyi Sun, Yanwei Li, Tao Zhang, Xueqing Deng, Henghui Ding, Lu Qi, Anran Wang, Xiangtai Li, Ming-Hsuan Yang*

Object-level grounded reasoning built on Sa2VA. Ships **VRT-Bench** (evaluation) and **VRT-80k** (training data).

📂 [`projects/vrt_sa2va`](./projects/vrt_sa2va/README.md) · [📜 arXiv](https://arxiv.org/pdf/2512.05091) · [🏠 Page](https://harboryuan.github.io/visual-reasoning-tracer/) · [🤗 Data](https://huggingface.co/datasets/HarborYuan/VRT-Eval)

### 🧩 [SAMTok](./projects/samtok/README.md) — Representing Any Mask with Two Words (CVPR 2026)
*Yikang Zhou, Tao Zhang, Dengxian Gong, Yuanzheng Wu, Ye Tian, Haochen Wang, Haobo Yuan, Jiacong Wang, Lu Qi, Hao Fei, Anran Wang, Zhuochen Wang, Yujing Wang, Cheng Chen, Shunping Ji, Xiangtai Li*

A unified mask-token interface that lets any MLLM generate and understand masks.

📂 [`projects/samtok`](./projects/samtok/README.md) · [📜 arXiv](https://arxiv.org/abs/2601.16093) · [🏠 Page](https://zhouyiks.github.io/projects/SAMTok/) · [🤗 Models](https://huggingface.co/collections/zhouyik/samtok)

### Extensions
- **[SaSaSa2VA](./projects/sasasa2va/README.md)** — a segmentation-augmented extension of Sa2VA that won **1st place** in the ICCV 2025 LSVOS Challenge RVOS Track 🏅.
- **[Pixel-SAIL](./projects/pixel_sail/README.md)** — single-transformer pixel-level grounding.

## Environment

We manage dependencies with [`uv`](https://docs.astral.sh/uv/). Install it once:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

The environment is defined under [`projects/sa2va`](./projects/sa2va) (`pyproject.toml` + `uv.lock`) and shared across the projects. The quickest way to set it up — with the virtualenv placed in `/tmp` and symlinked back into the project — is the helper script at the repo root:
```bash
bash setup_env.sh                 # projects/sa2va, --extra=latest
# bash setup_env.sh sa2va legacy  # InternVL2.5 or earlier
```
Or do it manually:
```bash
cd projects/sa2va
uv sync --extra=latest            # or --extra=legacy
```
Then run training / evaluation from the repository root with the environment activated (`source projects/sa2va/.venv/bin/activate`). See each project's README for project-specific steps.

For tokens / API keys (HuggingFace, OpenRouter), copy the template and fill it in — `setup_env.sh` loads it automatically:
```bash
cp .env.example .env   # then edit .env
```

Why `uv`? It treats the environment as code: dependencies are declared in `pyproject.toml` and every transitive package is version-locked in `uv.lock`. The result is a single source of truth that is fully reproducible across machines, trivial to maintain, and recreated exactly with one `uv sync` — no manual `pip install` drift.

## Citation

If you find this repository useful, please consider citing the relevant papers:

```bibtex
@article{sa2va,
  title={Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos},
  author={Yuan, Haobo and Li, Xiangtai and Zhang, Tao and Sun, Yueyi and Huang, Zilong and Xu, Shilin and Ji, Shunping and Tong, Yunhai and Qi, Lu and Feng, Jiashi and Yang, Ming-Hsuan},
  journal={arXiv pre-print},
  year={2025}
}

@article{yuan2025vrt,
  title={Visual Reasoning Tracer: Object-Level Grounded Reasoning Benchmark},
  author={Yuan, Haobo and Sun, Yueyi and Li, Yanwei and Zhang, Tao and Deng, Xueqing and Ding, Henghui and Qi, Lu and Wang, Anran and Li, Xiangtai and Yang, Ming-Hsuan},
  journal={arXiv pre-print},
  year={2025}
}

@inproceedings{zhou2026samtok,
  title={SAMTok: Representing Any Mask with Two Words},
  author={Zhou, Yikang and Zhang, Tao and Gong, Dengxian and Wu, Yuanzheng and Tian, Ye and Wang, Haochen and Yuan, Haobo and Wang, Jiacong and Qi, Lu and Fei, Hao and Wang, Anran and Wang, Zhuochen and Wang, Yujing and Chen, Cheng and Ji, Shunping and Li, Xiangtai},
  booktitle={CVPR},
  address={Denver, CO, USA},
  year={2026}
}
```
