import json
import torch

from projects.ST.eval.utils import Summary, AverageMeter, intersectionAndUnionGPU, master_only
from projects.ST.eval.datasets.RES import rle_to_mask
import argparse
import os.path as osp
from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from PIL import Image
from projects.ST.eval.utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu
from pycocotools import mask as mask_utils
import tqdm
import copy
import math
import os
import numpy as np

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"

class STBench(torch.utils.data.Dataset):

    def __init__(
        self,
        image_folder,
        json_file,
    ):
        self.image_folder = image_folder
        with open(json_file, "r") as f:
            json_data = json.load(f)
        self.json_data = json_data
        datas = []
        for json_item in self.json_data:
            image_name = json_item["image_name"]
            object_annotations = json_item["object_annotations"]
            for object_annotation in object_annotations:
                _item = {
                    "image_name": image_name, "segmentation": object_annotation["segmentation"],
                    "caption": object_annotation["caption"],
                }
                datas.append(_item)
        self.datas = datas

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        json_data = self.datas[idx]
        image_path = os.path.join(self.image_folder, json_data["image_name"])
        question = "<image>\nCan you provide me with a detailed description of the region in the picture marked by <region>? Please provide as detailed as possible."
        segmentations = json_data["segmentation"]
        caption = json_data["caption"]
        prompt_ids = [0]
        question = question.replace("<region>", "<Prompt0>")
        prompt_masks = [mask_utils.decode(segmentations)]

        image = Image.open(image_path).convert('RGB')

        return {
            "image": image, "prompt_masks": prompt_masks,
            "question": question, "img_id": int(idx), "image_path": image_path,
            "prompt_ids": prompt_ids, "caption": caption,
        }

def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth_model', help='pth model file')
    parser.add_argument('--save_path', help='save path')
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

    image_folder = "/mnt/bn/zhnagtao-lq/xiangtai-mnt/internvl_yk/vlm_before_20241125/data/masa_sam_500k/"

    dataset = STBench(
        image_folder=image_folder,
        json_file="./STBench_Detailed_Caption.json"
    )
    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        prediction = {'img_id': data_batch['img_id'], 'gt_caption': data_batch['caption']}
        prediction['image_path'] = data_batch['image_path']
        del data_batch['caption']
        del data_batch['image_path']
        del data_batch['img_id']

        text = data_batch['question']
        del data_batch['question']

        _data_batch = copy.deepcopy(data_batch)
        _data_batch['text'] = text
        # print(_data_batch)
        pred_caption = model.predict_forward(**_data_batch)['prediction']
        pred_caption = pred_caption.replace("<Prompt0>", "").replace("<|endoftext|>", "").strip()
        print(pred_caption)
        prediction.update({'pred_caption': pred_caption})
        results.append(prediction)

    tmpdir = './dist_test_temp_res_' + "STBench_" + "_detailed_cap"
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        if not os.path.exists("./work_dirs/ST_jsons/"):
            os.mkdir("./work_dirs/ST_jsons/")
        with open(args.save_path, 'w') as f:
            json.dump(results, f)

if __name__ == '__main__':
    main()



