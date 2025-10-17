# Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos

[\[🏠 Sa2VA\]](https://lxtgh.github.io/project/sa2va)  [\[📜 arXiv\]](https://arxiv.org/abs/2501.04001) [\[🤗 HuggingFace\]](https://huggingface.co/collections/ByteDance/sa2va-model-zoo-677e3084d71b5f108d00e093) [\[🎥 Introduction\]]() [\[🧑‍💻 GitHub\]](https://github.com/magic-research/Sa2VA) [\[Gradio Demo (Ours internal: Sa2VA-4B)\]](https://5512470799b6b35fbc.gradio.live/) [\[Gradio Demo (By HuggingFace Offical)\]](https://huggingface.co/spaces/fffiloni/Sa2VA-simple-demo) [\[🤖 Replicate Demo\]](https://replicate.com/bytedance)


[**Haobo Yuan**](https://yuanhaobo.me/)<sup>1*</sup> · [**Xiangtai Li**](https://lxtgh.github.io/)<sup>2*&dagger;</sup> · [**Tao Zhang**](https://zhang-tao-whu.github.io/)<sup>2,3*</sup> · [**Zilong Huang**](http://speedinghzl.github.io/)<sup>2</sup> · [**Shilin Xu**](https://xushilin1.github.io/)<sup>4</sup> ·[**Shunping Ji**](https://scholar.google.com/citations?user=FjoRmF4AAAAJ&hl=en)<sup>3</sup> ·[**Yunhai Tong**](https://scholar.google.com/citations?user=T4gqdPkAAAAJ&hl=zh-CN)<sup>4</sup> ·

[**Lu Qi**](https://luqi.info/)<sup>2</sup> · [**Jiashi Feng**](https://scholar.google.com/citations?user=Q8iay0gAAAAJ&hl=en)<sup>2</sup> · [**Ming-Hsuan Yang**](https://faculty.ucmerced.edu/mhyang/)<sup>1</sup>

<sup>1</sup>UC Merced&emsp;&emsp;&emsp;&emsp;<sup>2</sup>ByteDance Seed&emsp;&emsp;&emsp;&emsp;<sup>3</sup>WHU&emsp;&emsp;&emsp;&emsp;<sup>4</sup>PKU

&dagger; project lead&emsp;* the first three authors equally contribute to the work.

![Teaser](assets/images/teaser.jpg)

## Opensource progress

- [x] Release Qwen2.5-VL related models.
- [x] Release Open-sourced training datasets.
- [x] Release Ref-SAM-v dataset.
- [x] Release evaluation code for each dataset. 
- [x] Release 1B,4B,8B, 26B model.
- [x] Release training code for 1b, 4b, 8b model.
- [x] Release inference and test code.
- [x] Release demo code. 


## Overview

This repository contains the code for the paper "Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos".

Sa2VA is the first unified model for the dense grounded understanding of both images and videos. Unlike existing multi-modal large language models, which are often limited to specific modalities and tasks, Sa2VA supports a wide range of image and video tasks, including referring segmentation and conversation, with minimal one-shot instruction tuning. Sa2VA combines SAM-2, a foundation video segmentation model, with LLaVA, an advanced vision-language model, and unifies text, image, and video into a shared LLM token space.


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

## 🤗 Gradio Demos

We provide a script that implements interactive chat using gradio, which requires installing `gradio`. You can try it to build a local chat interface quickly.
```shell
PYTHONPATH=. python projects/sa2va/gradio/app.py ByteDance/Sa2VA-4B
```

## 🚀 Quick Start

Our Sa2VA model is available on 🤗HuggingFace. With very few steps, you can try it with your own data. You can install the `demo/requirements.txt` to avoid training-only packages.


**Option1 - scripts:**

Supposing you have a folder (`PATH_TO_FOLDER`) that contains images of a video, you can use the following script to chat with the Sa2VA model or segment the objects in the videos.

```bash
python demo/demo.py PATH_TO_FOLDER --model_path ByteDance/Sa2VA-8B --work-dir OUTPUT_DIR --text "<image>Please describe the video content."
```

If the output contains the segmentation results, the results will be saved to `OUTPUT_DIR`.

**Option2 - Jupter Notebook:**

Please refer to `demo.ipynb`.

## 🎥 Demo

<details open>
<summary>Demo 1</summary>
Input Video (Source: La La Land 2016):

![Error](assets/videos/exp_1.gif)

Instruction: "Please segment the girl wearing the yellow dress."
</details>

<details open>
<summary>Demo 2</summary>
Input Video (Source: La La Land 2016):

![Error](assets/videos/exp_2.gif)

Instruction: "Please segment the main character."
</details>


<details open>
<summary>Demo 3</summary>
Input Video (Source: Internet):

![Error](assets/videos/apt_exp_1_all.gif)

Instruction: "Please segment the person wearing sun glasses."
</details>


<details open>
<summary>Demo 4</summary>
Input Video (Source: Internet):

![Error](assets/videos/apt_exp_2_all.gif)

Instruction: "Instruction: "Please segment the singing girl."
</details>

<details open>
<summary>Demo 5</summary>
Input Video:

![Error](assets/videos/gf_exp1.gif)

Instruction: "What is the atmosphere of the scene?"

Answer: "The scene has a dark and mysterious atmosphere, with the men dressed in suits and ties, and the dimly lit room."
</details>


## Training
<details open>
<summary>Installation</summary>

We provide two ways for installation. Using `uv` is recommended for a faster and more reliable setup.

**Option 1: Using `uv` (Recommended)**

First, install `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Then, create a virtual environment and sync the dependencies:
```bash
uv sync --extra=latest # or uv sync --extra=legacy for Sa2VA based on InternVL2/2.5
source .venv/bin/activate
```

**Option 2: Using `conda` and `pip`**

Deprecated.

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
|   ├── refclef
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
`sam_v_full` is the SA-V dataset, which is not included in the download link. You can download it from [here](https://ai.meta.com/datasets/segment-anything-video/).


<details open>
<summary>Training Script</summary>

Please run the following script to train using 8 GPUS, we suggest using at least 8 A100 GPUs:
```bash
bash tools/dist.sh train projects/sa2va/configs/sa2va_in30_8b.py 8
```
</details>

<details open>
<summary>Convert trained model to huggingface format</summary>

Please run the following script to convert:
```bash
python tools/convert_to_hf.py projects/sa2va/configs/sa2va_in30_8b.py --pth-model PATH_TO_PTH_MODEL --save-path PATH_TO_SAVE_FOLDER
```
</details>

## Evaluation

<details open>
<summary>Image/Video Referring Segmentation Evaluation</summary>

Please adopt the following script to test Sa2VA on video object segmentation benchmarks using 8 GPUS.

You can use the following command to evaluate Sa2VA on all segmentation benchmarks at once:
```bash
python projects/sa2va/evaluation/run_all_evals.py /path/to/SA2VA/model --gpus 8
```
or you can evaluate Sa2VA on single segmentation benchmark(such as ReVOS):
```bash
./projects/llava_sam2/evaluation/dist_test.sh projects/llava_sam2/evaluation/ref_vos_eval.py path-to-hf-model 8 --work-dir path-to-output
```

</details>

<details open>
<summary>Image/Video QA Evaluation</summary>

We use [sa2va_eval](https://github.com/zhang-tao-whu/sa2va_eval) (a modified version of [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)) for Image/Video Chat benchmark evaluation.

**Single-GPU Evaluation Example:**
```bash
python run.py --data MMBench_DEV_EN MME SEEDBench_IMG --model Sa2VA-1B --verbose
```

**Multi-GPU Evaluation Example:**
```bash
torchrun --nproc-per-node=8 run.py --data MMBench_DEV_EN SEEDBench_IMG MMStar AI2D_TEST MMMU_DEV_VAL ScienceQA_TEST --model Sa2VA-4B Sa2VA-8B --verbose
```
</details>





## References
If you find this repository useful, please consider referring to the following paper:
```
@article{sa2va,
  title={Sa2VA: Marrying SAM2 with LLaVA for Dense Grounded Understanding of Images and Videos},
  author={Yuan, Haobo and Li, Xiangtai and Zhang, Tao and Huang, Zilong and Xu, Shilin and Ji, Shunping and Tong, Yunhai and Qi, Lu and Feng, Jiashi and Yang, Ming-Hsuan},
  journal={arXiv},
  year={2025}
}
```
