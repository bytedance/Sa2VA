# Copyright (c) OpenMMLab. All rights reserved.
import random

import torch
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)

import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from PIL import Image

import cv2
import json
import os
import copy
from panopticapi.utils import rgb2id
import numpy as np

class PixelCapVisualPromptDataset:
    def __init__(self,
                 image_folder,
                 pano_rgb_folder,
                 pano_json,
                 ):
        self.image_folder = image_folder
        self.pano_rgb_folder = pano_rgb_folder
        self.pano_json = pano_json

        self.datas = self.read_pano_json()

    def __len__(self):
        return len(self.datas)

    def read_pano_json(self):
        with open(self.pano_json, 'r') as f:
            json_info = json.load(f)
        ret = []
        for ann in json_info["annotations"]:
            image_id = int(ann["image_id"])
            image_file = os.path.join(self.image_folder, os.path.splitext(ann["file_name"])[0] + ".jpg")
            label_file = os.path.join(self.pano_rgb_folder, ann["file_name"])
            segments_info = copy.deepcopy(ann["segments_info"])
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "pan_seg_file_name": label_file,
                    "segments_info": segments_info,
                }
            )
        return ret

    def _parse_annotations(self, ann_info):
        image_path = ann_info['file_name']

        pano_seg = Image.open(ann_info["pan_seg_file_name"]).convert('RGB')
        segment_infos = ann_info["segments_info"]
        pan_seg_gt = rgb2id(np.array(pano_seg))

        masks, descriptions = [], []
        # print(segment_infos)
        for segment_info in segment_infos:
            if len(segment_info["description"]) > 5:
                masks.append(pan_seg_gt == segment_info["id"])
                descriptions.append(segment_info["description"])

        return {'masks': masks, 'descriptions': descriptions, 'image_path': image_path}

    def prepare_data(self, index):
        data_dict = self.datas[index]
        data_dict = self._parse_annotations(data_dict)
        return data_dict

    def __getitem__(self, index):
        data = self.prepare_data(index)
        return data

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def remove_prefix(state_dict, prefix):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default="phi3_chat",
        help='Specify a prompt template')
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template',
        choices=SYSTEM_TEMPLATE.keys(),
        default=None,
        help='Specify a system template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=2048,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
        'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
        'tokens with probabilities that add up to top_p or higher are '
        'kept for generation.')
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    # if args.cfg_options is not None:
        # cfg.merge_from_dict(args.cfg_options)

    cfg.model.pretrained_pth = None

    model = BUILDER.build(cfg.model)

    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    # del state_dict['llm.base_model.model.model.tok_embeddings.weight']
    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    if False:
        pass
    else:
        if args.with_plugins is None:
            inner_thoughts_open = False
            calculate_open = False
            solve_open = False
            search_open = False
        else:
            assert args.prompt_template == args.system_template == 'moss_sft'
            from plugins import plugins_api
            inner_thoughts_open = True
            calculate_open = 'calculate' in args.with_plugins
            solve_open = 'solve' in args.with_plugins
            search_open = 'search' in args.with_plugins
            # pre-import for api and model preparation
            if calculate_open:
                from plugins import calculate  # noqa: F401
            if solve_open:
                from plugins import solve  # noqa: F401
            if search_open:
                from plugins import search  # noqa: F401


        model.cuda()
        model.eval()
        model.preparing_for_generation(metainfo={})

        pixel2cap_dataset = PixelCapVisualPromptDataset(
            image_folder="data/coco/Images/train2017",
            pano_rgb_folder="data/coco/Annotations/annotations/panoptic_train2017/",
            pano_json="data/pixel2cap/pix2cap_coco_train.json",
        )

        for i in range(len(pixel2cap_dataset)):
            data_item = pixel2cap_dataset[i]

            image_path = data_item['image_path']
            prompt_id = random.randint(0, 9)
            text_prompts = f"<image>\nCan you provide me with a detailed description of the region in the picture marked by <Prompt{prompt_id}>?" + " Please provide as detailed as possible."
            # text_prompts = f"<image>\nWhat is <Prompt{prompt_id}>? Please select from the candidate categories. The candidate categories: person, cake, chair, window, car, giraffe, tree, sky, others."
            ori_image = Image.open(image_path).convert('RGB')
            for mask, description in zip(data_item['masks'], data_item['descriptions']):
                input_dict = {
                    'text': text_prompts,
                    'image': ori_image,
                    'prompt_masks': [mask],
                    'prompt_ids': [prompt_id]
                }

                return_dict = model.predict_forward(**input_dict)
                print(f"Prompt {prompt_id}: ", text_prompts, '\n', 'GT: ', description, '\n' 'Answer----', return_dict['prediction'], "\n\n")

def show_mask_pred(image, masks, save_dir='./output.png'):
    from PIL import Image
    import numpy as np

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255)]

    masks = torch.stack(masks, dim=0).cpu().numpy()[0, :]
    _mask_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    print(masks.shape)
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
        _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
        _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]


    image = np.array(image)
    image = image * 0.5 + _mask_image * 0.5
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_dir)

    return

if __name__ == '__main__':
    main()
