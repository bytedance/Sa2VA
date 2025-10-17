# Copyright (c) OpenMMLab. All rights reserved.
import json

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
from projects.ST.eval.utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from projects.ST.eval.datasets import RESDataset
from pycocotools import mask as _mask
import tqdm
import copy
import math
import os
import numpy as np

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
    parser.add_argument('--pth_model', help='pth model file')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
        default='refcoco',
        help='Specify a ref dataset')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')

    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco_plus'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
    'grefcoco': {'splitBy': "unc", 'dataset_name': 'grefcoco'},
}
IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'
DATA_PATH = './data/ref_seg/'

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
    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

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

    model.cuda()
    model.eval()
    model.preparing_for_generation(metainfo={})

    dataset_info = DATASETS_ATTRIBUTES[args.dataset]
    dataset = RESDataset(
        image_folder=IMAGE_FOLDER,
        dataset_name=dataset_info['dataset_name'],
        data_path=DATA_PATH,
        split=args.split,
        return_img_path=True,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        prediction = {'img_id': data_batch['img_id'], 'gt_masks': data_batch['gt_masks']}
        prediction['gt_masks'] = mask_to_rle(prediction['gt_masks'].cpu().numpy())
        prediction['image_path'] = data_batch['image_path']
        del data_batch['image_path']
        del data_batch['img_id'], data_batch['gt_masks']

        texts = data_batch['text']
        del data_batch['text']
        pred_masks = []
        for text in texts:
            _data_batch = copy.deepcopy(data_batch)
            _data_batch['text'] = text
            pred_mask = model.predict_forward(**_data_batch)['prediction_masks']
            if pred_mask is None or len(pred_mask) == 0:
                # give a zero mask
                print("No seg pred !!!")
                pred_masks.append(None)
            else:
                _ret_mask = pred_mask[0].cpu().numpy()
                _ret_mask = mask_to_rle(_ret_mask)
                pred_masks.append(_ret_mask)

        prediction.update({'prediction_masks': pred_masks})
        results.append(prediction)

    tmpdir = './dist_test_temp_res_' + args.dataset + args.split + "ST"
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        if not os.path.exists("./work_dirs/ST_jsons/"):
            os.mkdir("./work_dirs/ST_jsons/")
        with open(os.path.join("./work_dirs/ST_jsons/", f"{dataset_info['dataset_name']}.json"), 'w') as f:
            json.dump(results, f)
        metric = dataset.evaluate(results, './work_dirs')
        print(metric)

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

if __name__ == '__main__':
    main()
