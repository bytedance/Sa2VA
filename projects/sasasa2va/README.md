# SaSaSa2VA: Segmentation Augmented and Selective Averaged Sa2VA

[\[📜 arXiv\]](https://arxiv.org/abs/2509.16972) [\[🎥 YouTube\]](https://youtu.be/csBy0GYyvFw) [\[🧑‍💻 GitHub\]](https://github.com/bytedance/Sa2VA) [\[🤗 HuggingFace\]](https://huggingface.co/collections/QuanzhuNiu/sasasa2va-model-zoo-68de87ca8b59e75027003465) [\[🎯 Challenge\]](https://lsvos.github.io)


[**Quanzhu Niu**](https://scholar.google.com/citations?user=uX3u2Q0AAAAJ)<sup>1*</sup> · [**Dengxian Gong**](https://scholar.google.com/citations?user=51nzC3EAAAAJ&oi=ao)<sup>1*</sup> · [**Shihao Chen**](https://scholar.google.com/citations?user=W3fZBDQAAAAJ)<sup>1*</sup> · [**Tao Zhang**](https://zhang-tao-whu.github.io/)<sup>1*</sup> · [**Yikang Zhou**](https://zhouyiks.github.io)<sup>1</sup> · [**Haobo Yuan**](https://yuanhaobo.me/)<sup>2</sup> · [**Lu Qi**](https://luqi.info/)<sup>1</sup> · [**Xiangtai Li**](https://lxtgh.github.io/)<sup>3</sup> · [**Shunping Ji**](https://scholar.google.com/citations?user=FjoRmF4AAAAJ&hl=en)<sup>1&dagger;</sup>

<sup>1</sup>WHU&emsp;&emsp;&emsp;&emsp;<sup>2</sup>UC Merced&emsp;&emsp;&emsp;&emsp;<sup>3</sup>NTU

*equal contribution&emsp;&dagger;corresponding author

## 🎉 1st Place in ICCV 2025 LSVOS Challenge RVOS Track! 🎉

We win 1st place in ICCV 2025 LSVOS (Large-scale Video Object Segmentation) challenge RVOS (Referring Video Object Segmentation) track. The top 3 teams' methods are all based on Sa2VA. The challenge leaderborad:

| Method/Team Name |                             J\&F                             |                                                        Report                       |
|:----------:|:-----------------------------------------------------------------:|:----------------------------------------------------:|
| 🏅 SaSaSa2VA (Ours)  | **67.45** |     [📜 arXiv](https://arxiv.org/abs/2509.16972)  |
|  🥈 Transsion  | 64.65 |[📜 arXiv](https://arxiv.org/abs/2509.15546) |  
|  🥉 dytino | 64.14 |   [📜 arXiv](https://arxiv.org/abs/2509.19082) |

Please check out the full report of the challenge [here](https://arxiv.org/abs/2510.11063?).

### Our report video in ICCV 2025 has been released!

⬇️ Please click the teaser to watch the video. 

[![Watch our video!](https://img.youtube.com/vi/csBy0GYyvFw/maxresdefault.jpg)](https://www.youtube.com/watch?v=csBy0GYyvFw)

## Model Zoo

We provide the following models:
| Model Name |                             Base MLLM                             |                                                        HF Link                        |
|:----------:|:-----------------------------------------------------------------:|:----------------------------------------------------:|
|  SaSaSa2VA-4B  | [InternVL2.5-4B](https://huggingface.co/OpenGVLab/InternVL2_5-4B) |     [🤗 link](https://huggingface.co/QuanzhuNiu/SaSaSa2VA-4B) |
|  SaSaSa2VA-14B  | [InternVL3.5-14B](https://huggingface.co/OpenGVLab/InternVL3_5-14B) |   To be released |
|  SaSaSa2VA-26B | [InternVL2.5-26B](https://huggingface.co/OpenGVLab/InternVL2_5-26B) |   [🤗 link](https://huggingface.co/QuanzhuNiu/SaSaSa2VA-26B) |

## Usage

### Installation and Data Preparation

Please refer to [README.md](../../README.md) for installation and data preparation.

* If you use ```torch>=2.6.0```, there may be a problem about loading pth weight with ```xtuner==0.1.23```. We recommend using ```transformers==4.42.3``` for 4B/26B and ```transformers==4.52.1``` for 14B.
* If you use q_frame inference mode, you should generate frame indices by Q-frame or download from [🤗 Huggingface](https://huggingface.co/datasets/QuanzhuNiu/MeViS-Qframe). Then place them in `data/mevis_q_frame/valid/selected_frames.json` and `data/mevis_q_frame/valid_u/selected_frames.json`.

### Training

You can train SaSaSa2VA with:
```bash
bash tools/dist.sh train projects/sasasa2va/configs/YOUR_CONFIG NUM_GPUS
```
* Note: SaSaSa2VA uses pretrained weights of Sa2VA. Please put Sa2VA pretrained pth weights to `./pretrained/Sa2VA_pth/`.

Then run the following command to convert pth model to Huggingface format model:
```bash
python projects/sasasa2va/hf/convert_to_hf.py projects/sasasa2va/configs/YOUR_CONFIG --pth-model PATH_TO_PTH_MODEL --save-path PATH_TO_HF_MODEL
```

### Evaluation

You can evaluate SaSaSa2VA on MeViS valid (MEVIS) and valid_u (MEVIS_U) splits.

For inference:
```bash
projects/sasasa2va/evaluation/dist_test.sh projects/sasasa2va/evaluation/ref_vos_eval.py PATH_TO_HF_MODEL NUM_GPUS --dataset SPLIT --work-dir PATH_TO_OUTPUT --mode INFERENCE_MODE [--submit]
```
* We provide 5 inference modes: uniform(default), uniform_plus, q_frame, wrap_around, wrap_around_plus. If you use q_frame mode, please prepare q_frame indices in `data/mevis_q_frame/valid/selected_frames.json` and `data/mevis_q_frame/valid_u/selected_frames.json`.
* If you turn on `--submit`, the outputs will be .png format masks for [MeViS valid server](https://codalab.lisn.upsaclay.fr/competitions/15094).
* You can use `tools/llava_sam2_eval/eval_mevis.py` to compute metrics on MeViS valid_u split.

## Citation
If you find our work useful, please consider referring to the challenge report:
```
@article{sasasa2va,
  title={The 1st Solution for 7th LSVOS RVOS Track: {SaSaSa2VA}},
  author={Niu, Quanzhu and Gong, Dengxian and Chen, Shihao and Zhang, Tao and Zhou, Yikang and Yuan, Haobo and Qi, Lu and Li, Xiangtai and Ji, Shunping},
  journal={arXiv preprint arXiv:2509.16972},
  year={2025}
}

@article{liu2025lsvos,
  title={LSVOS 2025 Challenge Report: Recent Advances in Complex Video Object Segmentation},
  author={Chang Liu and Henghui Ding and Kaining Ying and Lingyi Hong and Ning Xu and Linjie Yang and Yuchen Fan and Mingqi Gao and Jingkun Chen and Yunqi Miao and Gengshen Wu and Zhijin Qin and Jungong Han and Zhixiong Zhang and Shuangrui Ding and Xiaoyi Dong and Yuhang Zang and Yuhang Cao and Jiaqi Wang and Chang Soo Lim and Joonyoung Moon and Donghyeon Cho and Tingmin Li and Yixuan Li and Yang Yang and An Yan and Leilei Cao and Feng Lu and Ran Hong and Youhai Jiang and Fengjie Zhu and Yujie Xie and Hongyang Zhang and Zhihui Liu and Shihai Ruan and Quanzhu Niu and Dengxian Gong and Shihao Chen and Tao Zhang and Yikang Zhou and Haobo Yuan and Lu Qi and Xiangtai Li and Shunping Ji and Ran Hong and Feng Lu and Leilei Cao and An Yan and Alexey Nekrasov and Ali Athar and Daan de Geus and Alexander Hermans and Bastian Leibe},
  journal={arXiv preprint arXiv:2510.11063},
  year={2025}
}
```