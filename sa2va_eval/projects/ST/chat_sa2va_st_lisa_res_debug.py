# Copyright (c) OpenMMLab. All rights reserved.
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

image_prefix = "./data/lisa_reasoning/val/"

demo_items = [
    {'image_path': image_prefix + '9263255964_46c5a14a8e_o.jpg', 'question': '<image>\nPlease segment where to wash hands.'},
    {'image_path': image_prefix + '9349045992_d06f09cc8e_o.jpg', 'question': '<image>\nPlease segment In order to facilitate transportation and connect different regions, what structure in the picture was built across the water?'},
    {'image_path': image_prefix + '4537339_3d8e488294_o.jpg', 'question': '<image>\nPlease segment something that the persons use to cross the water.'},
    {'image_path': image_prefix + '7657751362_9b18293c1c_o.jpg', 'question': '<image>\nPlease segment the dog that puts its tongue out.'},
    {'image_path': image_prefix + '457068560_f0e6b2a62e_o.jpg', 'question': '<image>\nPlease segment In an orchestra, musicians play different instruments. What object in the picture is commonly played with a bow to produce sound?'},
]

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

def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    frames = []

    frame_id = 0
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

        frame_id += 1

    cap.release()
    return frames

def get_frames_from_video(video_path, n_frames=5):
    frames = get_video_frames(video_path)
    stride = len(frames) / (n_frames + 1e-4)
    ret = []
    for i in range(n_frames):
        idx = int(i * stride)
        frame = frames[idx]
        frame = frame[:, :, ::-1]
        frame_image = Image.fromarray(frame).convert('RGB')
        ret.append(frame_image)
    return ret

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth_model', help='pth model file', default=None)

    parser.add_argument('--image', default=None, help='image')
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


def get_input():
    """Helper function for getting input from users."""
    sentinel = ''  # ends when this string is seen
    result = None
    while result is None:
        print(('\ndouble enter to end input (EXIT: exit chat, '
               'RESET: reset history) >>> '),
              end='')
        try:
            result = '\n'.join(iter(input, sentinel))
        except UnicodeDecodeError:
            print('Invalid characters detected. Please enter again.')
    return result


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

    if args.pth_model is not None:
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

        # save pth
        # state_dict = model.state_dict()
        # torch.save(state_dict, "pretrained/eve/eve.pth")

        for i, demo_item in enumerate(demo_items):
            image_path = demo_item['image_path']
            text_prompts = demo_item['question']
            # There is a video
            if '.mp4' in image_path:
                ori_image = get_frames_from_video(image_path, n_frames=5)
                ori_image_size = ori_image[0].size
                input_dict = {
                    'pixel_values': None,
                    'text_prompts': text_prompts,
                    'ori_image': ori_image,
                    'ori_image_size': ori_image_size,
                    'mode': 'demo_video',
                    'masks': None
                }
            else:
                ori_image = Image.open(image_path).convert('RGB')
                ori_image_size = ori_image.size
                input_dict = {
                    'text': text_prompts,
                    'image': ori_image,
                }

            return_dict = model.predict_forward(**input_dict)
            print(i, ': ', return_dict['prediction'])
            if 'prediction_masks' in return_dict.keys() and return_dict['prediction_masks'] is not None:
                show_mask_pred(ori_image, return_dict['prediction_masks'], save_dir=f'./demos/output_{i}.png')

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
