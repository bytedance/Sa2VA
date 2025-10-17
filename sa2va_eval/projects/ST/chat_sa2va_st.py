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

sam_prefix = '/mnt/bn/zhnagtao-lq/xiangtai-mnt/sam/sa_000001/'
coco_prefix = 'data/glamm_data/images/coco2014/train2014/'
sam_p2 = 'data/sa_eval/'
coco_test_prefix = "/mnt/bn/zhnagtao-lq/xiangtai-mnt/tt_vlm/data/coco/Images/val2017/"

demo_items = [
    # {'image_path': './work_dirs/demo_figs/french-bulldog-8163486_1280.jpg', 'question': '<image>\nPlease detailed describe the image.'},
    # {'image_path': './work_dirs/demo_figs/sunset-8064078_1280.jpg', 'question': '<image>\nPlease detailed describe the image.'},
    # {'image_path': './work_dirs/demo_figs/traditional-8503473_1280.jpg', 'question': '<image>\nPlease detailed describe the image.'},
    # {'image_path': './work_dirs/demo_figs/traditional-8503473_1280.jpg', 'question': '<image>\nWhat the women is doing?'},
    # {'image_path': './work_dirs/demo_figs/lemon-cake-8274419_1280.jpg', 'question': '<image>\nPlease segment the sourest thing in the picture.'},
    # {'image_path': './work_dirs/demo_figs/canoe-7541311_1280.jpg', 'question': '<image>\nPlease detailed describe the man.'},
    # {'image_path': './work_dirs/demo_figs/canoe-7541311_1280.jpg', 'question': '<image>\nPlease segment the tool that the man uses to push the boat.'},
    # {'image_path': './work_dirs/demo_figs/canoe-7541311_1280.jpg', 'question': '<image>\nPlease segment what is supporting the man to keep him afloat on the water.'},
    # {'image_path': './work_dirs/demo_figs/canoe-7541311_1280.jpg', 'question': '<image>\n.If the man accidentally falls into the water, what in the image will help him avoid drowning?'},
    # {'image_path': './work_dirs/demo_figs/hut-8843868_1280.jpg', 'question': '<image>\n.Please segment the house.'},
    # {'image_path': './work_dirs/demo_figs/hut-8843868_1280.jpg', 'question': '<image>\n.Please segment the reflection of the house in the water.'},
    # {'image_path': './work_dirs/demo_figs/ai-generated-8637800_1280.jpg', 'question': '<image>\nWhat is unusual about this picture?'},

    # {'image_path': './work_dirs/demo_figs/spaghetti-6639970_1280.jpg', 'question': '<image>\nPlease segment the cooker.'},
    # {'image_path': './work_dirs/demo_figs/spaghetti-6639970_1280.jpg', 'question': '<image>\nPlease segment the cooked pasta.'},
    #
    # {'image_path': './work_dirs/demo_figs/bmx-5142643_1280.jpg', 'question': '<image>\nPlease segment the cameraman in the image.'},
    # {'image_path': './work_dirs/demo_figs/bmx-5142643_1280.jpg', 'question': '<image>\nPlease segment the person who is riding the bicycle.'},
    # {'image_path': './work_dirs/demo_figs/bmx-5142643_1280.jpg', 'question': '<image>\nPlease segment the the bicycle.'},
    #
    # {'image_path': './work_dirs/demo_figs/car-7862030_1280.jpg', 'question': '<image>\nPlease segment the car nearest the camera.'},
    #
    # {'image_path': './work_dirs/demo_figs/pham-ngu-lao-3989110_1280.jpg', 'question': '<image>\nPlease segment the red electric motorcycle ridden by a man.'},
    # {'image_path': './work_dirs/demo_figs/pham-ngu-lao-3989110_1280.jpg', 'question': '<image>\nPlease segment the trash can with "E14".'},
    # {'image_path': './work_dirs/demo_figs/pham-ngu-lao-3989110_1280.jpg', 'question': '<image>\nPlease segment the garbage bags.'},
    # {'image_path': sam_prefix+'sav_000003.mp4', 'question': '<image>\nPlease describe the video.'},
    # {'image_path': sam_prefix+'sav_000003.mp4', 'question': '<image>\nHow many dogs in the video?'},
    # {'image_path': sam_prefix+'sav_000004.mp4', 'question': '<image>\nHow many handbags is brought by the man?'},
    # {'image_path': sam_prefix+'sav_000001.mp4', 'question': '<image>\nWhat the child is doing?'},
    # {'image_path': sam_prefix+'sav_000021.mp4', 'question': '<image>\nPlease describe the video.'},
    # {'image_path': sam_prefix+'sav_000039.mp4', 'question': '<image>\nIs the red car in the video moving or stationary?'},
    # {'image_path': sam_prefix+'sav_000042.mp4', 'question': '<image>\nPlease describe the man\'s actions in the video.'},


    # {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease describe the image.'},
    # {'image_path': sam_prefix+'sa_21340.jpg', 'question': '<image>\nPlease describe the image.'},
    # {'image_path': sam_prefix+'sa_21340.jpg', 'question': '<image>\nCan you segment the bucket in this image? Segmentation in the instance mode.'},
    # {'image_path': sam_prefix+'sa_21340.jpg', 'question': '<image>\nCan you segment the bucket in this image? Segmentation in the semantic mode.'},
    # {'image_path': sam_prefix+'sa_21340.jpg', 'question': '<image>\nPlease segment the largest bucket.'},
    # {'image_path': sam_prefix+'sa_21340.jpg', 'question': '<image>\nPlease segment the smallest bucket.'},
    # {'image_path': sam_prefix+'sa_21340.jpg', 'question': '<image>\nPlease segment the blue bucket.'},
    # {'image_path': sam_prefix+'sa_20341.jpg', 'question': '<image>\nPlease describe the image.'},

    {'image_path': './test.jpg', 'question': '<image>\nPlease describe this picture.'},
    {'image_path': 'demos/1689142393914482.png', 'question': '<image>\nPlease segment the squirrel. Segmentation in the instance mode.'},
    {'image_path': 'demos/1689142393914482.png', 'question': '<image>\nPlease segment the squirrel face to the left.'},
    {'image_path': 'demos/1689142393914482.png', 'question': '<image>\nPlease segment the right squirrel.'},
    {'image_path': 'demos/1689142393914482.png', 'question': '<image>\nPlease segment the head of the left squirrel.'},
    {'image_path': 'demos/1689142393914482.png', 'question': '<image>\nPlease segment the hands of the left squirrel.'},
    # {'image_path': coco_test_prefix + '000000001000.jpg', 'question': '<image>\nPlease segment the man in red shirt.'},
    # {'image_path': coco_test_prefix + '000000001000.jpg', 'question': '<image>\nPlease segment the person. Segmentation in the instance mode.'},
    # {'image_path': coco_test_prefix + '000000001000.jpg', 'question': '<image>\nCould you please give me a brief description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.'},

    # {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the snowboarder.'},
    # {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the snowboard.'},
    # {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the person.'},
    # {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the forest.'},
    # {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the forest. Segmentation in the semantic mode.'},
    # {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the person. Segmentation in the instance mode.'},
    # {'image_path': coco_prefix+'COCO_train2014_000000581921.jpg', 'question': '<image>\nPlease segment the snow filed. Segmentation in the semantic mode.'},

    # {'image_path': coco_prefix + 'COCO_train2014_000000000025.jpg', 'question': '<image>\nWhat kind of animal is in the picture?'},
    # {'image_path': coco_prefix + 'COCO_train2014_000000000025.jpg',
    #  'question': '<image>\nWhat is the giraffe doing?'},
    # {'image_path': coco_prefix + 'COCO_train2014_000000000025.jpg', 'question': '<image>\nCan you segment the giraffe in this image? Segmentation in the instance mode.'},
    # {'image_path': coco_prefix + 'COCO_train2014_000000000025.jpg', 'question': '<image>\nCan you segment the giraffe in this image? Segmentation in the semantic mode.'},
    # {'image_path': coco_prefix + 'COCO_train2014_000000000025.jpg', 'question': '<image>\nCan you segment the tree in this image? Segmentation in the instance mode.'},
    # {'image_path': sam_p2+'sav_053576.mp4', 'question': '<image>\nPlease describe the video.'}
    # {'image_path': sam_p2+'sav_053474.mp4', 'question': '<image>\nWhat is the weather now?'},
    # {'image_path': sam_p2 + 'sav_053474.mp4', 'question': '<image>\nWhat is the speed limit in this road?'},
    # {'image_path': sam_p2 + 'sav_053474.mp4', 'question': '<image>\nWhat is the color of the front car?'},
    # {'image_path': "./1215_demos/qilu.jpg", 'question': '<image>\nPlease detailed describe the image.'},
    # {'image_path': "./1215_demos/qilu.jpg", 'question': '<image>\nPlease detailed describe the image and response with segmentation masks.'},
    # {'image_path': sam_p2 + "sora_tokyo_walk.mp4", 'question': '<image>\nWhich country do you think this is?'},
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
