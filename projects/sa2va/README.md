# Sa2VA: Marrying SAM2 with MLLM for Dense Grounded Understanding of Images and Videos (IEEE TPAMI 2026)

[\[🏠 Sa2VA\]](https://lxtgh.github.io/project/sa2va)  [\[📕 TPAMI\]](https://ieeexplore.ieee.org/document/11640960) [\[📜 arXiv\]](https://arxiv.org/abs/2501.04001) [\[🤗 HuggingFace\]](https://huggingface.co/collections/ByteDance/sa2va-model-zoo-677e3084d71b5f108d00e093) [\[Gradio Demo (HuggingFace Offical)\]](https://huggingface.co/spaces/fffiloni/Sa2VA-simple-demo) [\[🤖 Replicate Demo\]](https://replicate.com/bytedance)


[**Haobo Yuan**](https://yuanhaobo.me/)<sup>1*</sup> · [**Xiangtai Li**](https://lxtgh.github.io/)<sup>2*&dagger;</sup> · [**Tao Zhang**](https://zhang-tao-whu.github.io/)<sup>2,3*</sup> · [**Yueyi Sun**]()<sup>4</sup> · [**Zilong Huang**](http://speedinghzl.github.io/)<sup>2</sup> · [**Shilin Xu**]()<sup>4</sup> ·[**Shunping Ji**](https://scholar.google.com/citations?user=FjoRmF4AAAAJ&hl=en)<sup>3</sup> ·[**Yunhai Tong**](https://scholar.google.com/citations?user=T4gqdPkAAAAJ&hl=zh-CN)<sup>4</sup> · [**Lu Qi**](https://luqi.info/)<sup>3</sup> · [**Jiashi Feng**](https://scholar.google.com/citations?user=Q8iay0gAAAAJ&hl=en)<sup>2</sup> · [**Ming-Hsuan Yang**](https://faculty.ucmerced.edu/mhyang/)<sup>1</sup>

<sup>1</sup>UC Merced&emsp;&emsp;&emsp;&emsp;<sup>2</sup>ByteDance Seed&emsp;&emsp;&emsp;&emsp;<sup>3</sup>WHU&emsp;&emsp;&emsp;&emsp;<sup>4</sup>PKU

&dagger; project lead&emsp;* the first three authors equally contribute to the work.

> Part of the [Sa2VA repository](../../README.md). See the top-level README for the full project family (VRT, SAMTok, SaSaSa2VA).

![Teaser](../../assets/images/teaser.jpg)

## News

- **[2026-07-28]** 🎉 Sa2VA is accepted to **[IEEE TPAMI 2026](https://ieeexplore.ieee.org/document/11640960)**!
- **[2026-06-15]** Added [Sa2VA-LLaVA-1.5-7B](https://huggingface.co/ByteDance/Sa2VA-LLaVA-1.5-7B), a LLaVA-1.5-7B (CLIP-ViT-L-336 + Vicuna-7B) variant with a SAM2 grounding encode.
- **[2026-06-10]** Added [Sa2VA-Qwen3-VL-4B-SAM3](https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-4B-SAM3), a Qwen3-VL-4B variant with a SAM3 grounding encoder.

## Overview

Sa2VA is the first unified model for the dense grounded understanding of both images and videos. Unlike existing multi-modal large language models, which are often limited to specific modalities and tasks, Sa2VA supports a wide range of image and video tasks, including referring segmentation and conversation, with minimal one-shot instruction tuning. Sa2VA combines SAM-2, a foundation video segmentation model, with an advanced multimodal LLM (MLLM), and unifies text, image, and video into a shared LLM token space.

Sa2VA produces segmentation masks by emitting a special `[SEG]` token from the MLLM; its hidden state is projected into SAM-2's prompt space, which decodes the corresponding mask(s). This single mechanism powers image/video referring segmentation, grounded conversation generation (GCG), and visual prompting, while standard image/video chat is handled by the underlying MLLM. Sa2VA supports multiple MLLM backbones — InternVL2.5, InternVL3, Qwen2.5-VL, and Qwen3-VL.

### Tasks at a Glance

- **Referring segmentation** — segment objects in images/videos from a free-form language expression (RefCOCO/+/g, ReVOS, MeViS, DAVIS, Ref-SAV).
- **Grounded conversation generation (GCG)** — generate captions with inline `[SEG]` masks grounding the mentioned entities.
- **Visual prompting** — answer questions about user-specified regions.
- **Image & video chat / QA** — general multimodal conversation and video understanding.

## Model Zoo

We provide the following models:
| Model Name |                             Base MLLM                             |                                 Language Part                                 |                       HF Link                        |
|:----------:|:-----------------------------------------------------------------:|:-----------------------------------------------------------------------------:|:----------------------------------------------------:|
|  Sa2VA-1B  | [InternVL2.5-1B](https://huggingface.co/OpenGVLab/InternVL2_5-1B) |   [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)    | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-1B) |
|  Sa2VA-4B  | [InternVL2.5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B) |    [Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)     | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-4B) |
|  Sa2VA-8B  | [InternVL2.5-8B](https://huggingface.co/OpenGVLab/InternVL2_5-8B) |  [internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-8B) |
|  Sa2VA-26B | [InternVL2.5-26B](https://huggingface.co/OpenGVLab/InternVL2_5-26B) |  [internlm2_5-20b-chat](https://huggingface.co/internlm/internlm2_5-20b-chat)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-26B) |
|  Sa2VA-InternVL3-2B	 | [InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B) |  [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-InternVL3-2B) |
|  Sa2VA-InternVL3-8B	 | [InternVL3-8B](https://huggingface.co/OpenGVLab/InternVL3-8B) |  [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-InternVL3-8B) |
|  Sa2VA-InternVL3-14B	 | [InternVL3-14B](https://huggingface.co/OpenGVLab/InternVL3-14B) |  [Qwen2.5-14B](https://huggingface.co/Qwen/Qwen2.5-14B)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-InternVL3-14B) |
|  Sa2VA-Qwen2_5-VL-3B	 | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |  [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-Qwen2_5-VL-3B) |
|  Sa2VA-Qwen2_5-VL-7B	 | [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |  [Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-Qwen2_5-VL-7B) |
|  Sa2VA-Qwen3-VL-2B	 | [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) |  [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-2B) |
|  Sa2VA-Qwen3-VL-4B	 | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |  [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-4B) |
|  Sa2VA-Qwen3-VL-4B-SAM3	 | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) |  [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-Qwen3-VL-4B-SAM3) |
|  Sa2VA-LLaVA-1.5-7B	 | [LLaVA-1.5-7B](https://huggingface.co/llava-hf/llava-1.5-7b-hf) |  [Vicuna-7B](https://huggingface.co/lmsys/vicuna-7b-v1.5)   | [🤗 link](https://huggingface.co/ByteDance/Sa2VA-LLaVA-1.5-7B) |

## Environment

Use `uv` to manage dependencies. First install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
The environment is defined here in `projects/sa2va` (`pyproject.toml` + `uv.lock`). The easiest way to set it up is the helper script at the repo root, which puts the virtualenv in `/tmp` and symlinks it back into the project:
```bash
bash setup_env.sh sa2va latest    # or: bash setup_env.sh sa2va legacy
```

Or sync manually from this directory, choosing the extra based on your model family:
- `uv sync --extra=latest` for newer models — Qwen3-VL, Qwen2.5-VL, InternVL3 (latest Transformers).
- `uv sync --extra=legacy` for InternVL2.5 or earlier models (legacy Transformers).

```bash
cd projects/sa2va
uv sync --extra=latest   # or: uv sync --extra=legacy
source .venv/bin/activate
```
Run training / evaluation commands from the **repository root** (with this environment activated) — the code uses repo-root-relative imports such as `projects.sa2va`, `third_parts`, and `vlm`.

## 🚀 Quick Start

Our Sa2VA model is available on 🤗HuggingFace. With very few steps, you can try it with your own data. You can install the `projects/sa2va/demo/requirements.txt` to avoid training-only packages.

**Option 1 - scripts:**

Supposing you have a folder (`PATH_TO_FOLDER`) that contains images of a video, you can use the following script to chat with the Sa2VA model or segment the objects in the videos.

```bash
python projects/sa2va/demo/demo.py PATH_TO_FOLDER --model_path ByteDance/Sa2VA-8B --work-dir OUTPUT_DIR --text "<image>Please describe the video content."
```

If the output contains the segmentation results, the results will be saved to `OUTPUT_DIR`.

**Option 2 - Jupyter Notebook:**

Please refer to `projects/sa2va/demo.ipynb`.

**Option 3 - Gradio:**

We provide a script that implements interactive chat using gradio, which requires installing `gradio`. You can try it to build a local chat interface quickly.
```shell
PYTHONPATH=. python projects/sa2va/gradio/app.py ByteDance/Sa2VA-4B
```

## 🎥 Demo

<details open>
<summary>Demo 1</summary>
Input Video (Source: La La Land 2016):

![Error](../../assets/videos/exp_1.gif)

Instruction: "Please segment the girl wearing the yellow dress."
</details>

<details open>
<summary>Demo 2</summary>
Input Video (Source: La La Land 2016):

![Error](../../assets/videos/exp_2.gif)

Instruction: "Please segment the main character."
</details>


<details open>
<summary>Demo 3</summary>
Input Video (Source: Internet):

![Error](../../assets/videos/apt_exp_1_all.gif)

Instruction: "Please segment the person wearing sun glasses."
</details>


<details open>
<summary>Demo 4</summary>
Input Video (Source: Internet):

![Error](../../assets/videos/apt_exp_2_all.gif)

Instruction: "Please segment the singing girl."
</details>

<details open>
<summary>Demo 5</summary>
Input Video:

![Error](../../assets/videos/gf_exp1.gif)

Instruction: "What is the atmosphere of the scene?"

Answer: "The scene has a dark and mysterious atmosphere, with the men dressed in suits and ties, and the dimly lit room."
</details>


## Training
<details open>
<summary>Pretrained Model Preparation</summary>

You are expected to download the following pretrained models and place them in the `./pretrained` directory:
- [sam2_hiera_large.pt](https://huggingface.co/facebook/sam2-hiera-large)
- [InternVL2_5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B)

You can download the remaining models from InternVL2.5 [huggingface collections](https://huggingface.co/collections/OpenGVLab/internvl25-673e1019b66e2218f68d7c1c).

```
./ # project root
pretrained/
├── sam2_hiera_large.pt
├── InternVL2_5-1B
├── InternVL2_5-4B
```
</details>

<details open>
<summary>Data Preparation</summary>

Please download the training datasets and place them in the `data` directory. The download link is [here](https://huggingface.co/datasets/Dense-World/Sa2VA-Training).

Please directly put the zip files into the `data` directory and unzip them. For example, you can download the `video_datas_mevis.zip` and unzip it in the `data` directory like:
```bash
unzip video_datas_mevis.zip
```

The final data structure should be like:
```
data/
├── video_datas
|   ├── revos
|   ├── mevis
|   └── davis17
|   └── chat_univi
|   └── sam_v_full # [!important] please download this from sam-2 directly.
|   └── Ref-SAV.json
├── ref_seg
|   ├── refcoco
|   ├── refcoco+
|   ├── refcocog
|   ├── 
├── glamm_data
|   ├── images
|   ├── annotations
├── osprey-724k
|   ├── Osprey-724K
|   ├── coco
├── llava_data
|   ├── llava_images
|   ├── LLaVA-Instruct-150K
|   ├── LLaVA-Pretrain

```
**Important**: `sam_v_full` is the SA-V dataset, which is not included in the download link. You can download it from **Meta** ([here](https://ai.meta.com/datasets/segment-anything-video/)). Please follow their license.
</details>

<details open>
<summary>Training Script</summary>

Please run the following script to train using 8 GPUS, we suggest using at least 8 A100 GPUs:
```bash
bash tools/dist.sh train projects/sa2va/configs/sa2va_in30_8b.py 8
```

Configs for other backbones live under `projects/sa2va/configs/` (InternVL3: `sa2va_in30_*.py`; Qwen2.5-VL: `sa2va_qwenvl25/`; Qwen3-VL: `sa2va_qwenvl3/`).
</details>

<details open>
<summary>Fine-tuning</summary>

We provide a simple example for fine-tuning Sa2VA on an image referring segmentation task. For detailed instructions, please refer to our [fine-tuning guide](docs/finetune.md).

The example dataset is constructed from a few images from RefCOCO. To fine-tune on your own data, you can organize it in the same format as our example `annotations.json`. You can download the example dataset from [Hugging Face](https://huggingface.co/datasets/bitersun/Sa2VA-finetune-example).

For other types of data, you may need to customize the dataloader and configuration. Please refer to `projects/sa2va/datasets/sa2va_data_finetune.py` and `projects/sa2va/configs/sa2va_finetune.py` for guidance.
</details>

<details open>
<summary>Convert trained model to huggingface format</summary>

Please run the following script to convert:
```bash
python tools/convert_to_hf.py projects/sa2va/configs/sa2va_in30_8b.py --pth-model PATH_TO_PTH_MODEL --save-path PATH_TO_SAVE_FOLDER
```
</details>

## Evaluation

You can download Ref-SAV eval set [here🤗](https://huggingface.co/datasets/Dense-World/Sa2VA-Eval).

<details open>
<summary>Image/Video Referring Segmentation Evaluation</summary>

Please adopt the following script to test Sa2VA on video object segmentation benchmarks using 8 GPUS.

You can use the following command to evaluate Sa2VA on all segmentation benchmarks at once:
```bash
python projects/sa2va/evaluation/run_all_evals.py /path/to/SA2VA/model --gpus 8
```
or you can evaluate Sa2VA on single segmentation benchmark(such as ReVOS):
```bash
./projects/sa2va/evaluation/dist_test.sh projects/sa2va/evaluation/sa2va_eval_ref_vos.py path-to-hf-model 8 --work_dir path-to-output
```
</details>

<details open>
<summary>Image/Video QA Evaluation</summary>

We use [sa2va_eval](https://github.com/zhang-tao-whu/sa2va_eval) (a modified version of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)) for Image/Video Chat benchmark evaluation.

**Single-GPU Evaluation Example:**
```bash
python sa2va_eval/run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model Sa2VA-1B --verbose
```

**Multi-GPU Evaluation Example:**
```bash
torchrun --nproc-per-node=8 sa2va_eval/run.py --data MMBench_DEV_EN SEEDBench_IMG MMStar AI2D_TEST MMMU_DEV_VAL ScienceQA_TEST --model Sa2VA-4B Sa2VA-8B --verbose
```
</details>

## Citation
If you find this project useful, please consider citing:
```bibtex
@article{sa2va,
  title={Sa2VA: Marrying SAM2 with MLLM for Dense Grounded Understanding of Images and Videos},
  author={Yuan, Haobo and Li, Xiangtai and Zhang, Tao and Sun, Yueyi and Huang, Zilong and Xu, Shilin and Ji, Shunping and Tong, Yunhai and Qi, Lu and Feng, Jiashi and Yang, Ming-Hsuan},
  journal={IEEE TPAMI},
  year={2026}
}
```
