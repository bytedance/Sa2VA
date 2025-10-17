import re
import torch
from pycocotools.coco import COCO

import json
from pycocotools import mask as _mask
import tqdm
import math
import os
import numpy as np
import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from PIL import Image

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocog region caption')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth_model', help='pth model file')
    parser.add_argument(
        '--output-path',
        default='./region_cap_pred.json',
        help='save path of the prediction')
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

class RegionCapInferenceDataset:
    def __init__(self,
                 image_folder,
                 annotation_file=None,
                 ):
        self.image_folder = image_folder
        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())

    def __len__(self):
        return len(self.image_dict_keys)

    def decode_mask(self, annotation, image_info):
        flag = False
        masks = []

        for ann_id in range(1):

            ann = {"segmentation": annotation}

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = _mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = _mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
        masks = np.stack(masks, axis=0)
        return masks

    def get_questions(self):
        # question = "<image>\nPlease give me a short description of <Prompt0>."
        question = "<image>\nCan you provide me with a detailed description of the region in the picture marked by <Prompt0>?"
        return question

    def __getitem__(self, index):

        data_dict = {}

        image_id = self.image_dict_keys[index]
        image_file = self.image_dict[image_id]['file_name']

        questions = self.get_questions()

        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        masks = self.ann_dict[image_id]['segmentation']
        image_info = self.image_dict[image_id]
        masks = self.decode_mask(masks, image_info)

        data_dict['image'] = image
        data_dict['text'] = questions
        data_dict['img_id'] = image_id
        data_dict['prompt_masks'] = [masks[0]]
        data_dict['prompt_ids'] = [0]
        return data_dict


ANNOTATION_FILE = './data/region_caption/finetune_refcocog_val_with_mask.json'
IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'

def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
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

    dataset = RegionCapInferenceDataset(
        image_folder=IMAGE_FOLDER,
        annotation_file=ANNOTATION_FILE,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        result_dict = {'image_id': data_batch['img_id'], 'image_file': data_batch['image_file']}
        del data_batch['img_id'], data_batch['image_file']

        prediction = model.predict_forward(**data_batch)['prediction']

        text_output = prediction.replace("<|endoftext|>", "").replace(".", "").replace("<Prompt0>", "").replace(":", "").strip()
        print(text_output)
        text_output = text_output.split("ASSISTANT: ")[-1]
        cleaned_str = re.sub(r'<.*?>', '', text_output)
        cleaned_str = cleaned_str.replace('[SEG]', '')
        cleaned_str = ' '.join(cleaned_str.split()).strip("'")
        cleaned_str = cleaned_str.strip()

        result_dict["caption"] = cleaned_str
        result_dict["prediction"] = cleaned_str
        results.append(result_dict)

    tmpdir = './dist_test_temp_regioncap'
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        with open(args.output_path, 'w') as json_file:
            json.dump(results, json_file, indent=2)

if __name__ == '__main__':
    main()
