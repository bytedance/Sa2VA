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

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        json_data = self.json_data[idx]
        image_path = os.path.join(self.image_folder, json_data["image_name"])
        question = json_data["question"]
        answer = json_data["answer"]
        segmentations = json_data["segmentations"]

        prompt_ids = [i for i in range(len(segmentations))]

        replace_dict = {}
        replace_dict_id = {}
        for obj_tag, prompt_id in zip(segmentations.keys(), prompt_ids):
            replace_dict[obj_tag] = f"<Prompt{prompt_id}>"
            replace_dict_id[obj_tag] = prompt_id

        question_masks = []
        gt_masks = []
        input_parompt_ids = []
        for _key in replace_dict.keys():
            rle_dict = segmentations[_key]
            mask = mask_utils.decode(rle_dict)
            if _key in question:
                question = question.replace(_key, replace_dict[_key])
                question_masks.append(mask)
                input_parompt_ids.append(replace_dict_id[_key])
            elif _key in answer:
                gt_masks.append(mask)
            else:
                raise NotImplementedError

        image = Image.open(image_path).convert('RGB')

        gt_masks = np.stack(gt_masks, axis=0)
        gt_masks = torch.from_numpy(gt_masks)

        return {
            "image": image, "gt_masks": gt_masks,
            "question": question, "img_id": int(idx), "image_path": image_path,
            "prompt_masks": question_masks, "prompt_ids": input_parompt_ids,
        }

    @master_only
    def evaluate(self, result, work_dir):
        trackers = {
            "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union": AverageMeter("Union", ":6.3f", Summary.SUM),
            "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
        }
        for pred_dict in result:
            intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            masks = pred_dict['prediction_masks']
            _masks = []
            for mask in masks:
                if mask is not None:
                    mask = rle_to_mask(mask)
                _masks.append(mask)
            targets = pred_dict['gt_masks']
            _targets = rle_to_mask(targets)

            for i_item, _mask in enumerate(_masks):
                if _mask is None:
                    continue

                _target = _targets[i_item: i_item + 1]
                for prediction, target in zip(_mask, _target):
                    prediction = torch.from_numpy(prediction).int().cuda()
                    target = torch.from_numpy(target).int().cuda()
                    intersect, union_, _ = intersectionAndUnionGPU(
                        prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                    )
                    intersection += intersect
                    union += union_
                    accuracy_iou += intersect / (union_ + 1e-5)
                    accuracy_iou[union_ == 0] += 1.0
            if not isinstance(intersection, float):
                intersection = intersection.cpu().numpy()
            if not isinstance(union, float):
                union = union.cpu().numpy()
            # intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
            if not isinstance(accuracy_iou, float):
                accuracy_iou = accuracy_iou.cpu().numpy()
            accuracy_iou = accuracy_iou / _targets.shape[0]
            trackers["intersection"].update(intersection)
            trackers["union"].update(union)
            trackers["gIoU"].update(accuracy_iou, n=_targets.shape[0])

        cur_results = {'pixel_intersection': trackers["intersection"].sum[1],
                       'pixel_union': trackers["union"].sum[1],
                       'gIoU': trackers["gIoU"].avg[1],
                       'mask_counts': trackers["gIoU"].count,
                       }
        class_iou = cur_results['pixel_intersection'] / (cur_results['pixel_union'] + 1e-10)
        global_iou = cur_results['gIoU']

        print('============================================', 'current')
        print('CIoU: {}, GIoU: {}'.format(class_iou, global_iou), 'current')
        print('============================================', 'current')
        print('VRES successfully finished evaluating',
              'current')
        return {'Acc': class_iou}


def parse_args():
    parser = argparse.ArgumentParser(description='Chat with a HF model')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('--pth_model', help='pth model file')
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
        json_file="./STBench_VRES.json"
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

        text = data_batch['question']
        del data_batch['question']
        pred_masks = []
        _data_batch = copy.deepcopy(data_batch)
        _data_batch['text'] = "<image>\n" + text
        # print(_data_batch)
        pred_mask = model.predict_forward(**_data_batch)['prediction_masks']
        if pred_mask is None or len(pred_mask) == 0:
            # give a zero mask
            # print("No seg pred !!!")
            pred_mask = model.predict_forward(**_data_batch, prefix_answer="It is [SEG].")['prediction_masks']
            if pred_mask is None or len(pred_mask) == 0:
                print("No seg pred !!!")
                pred_masks.append(None)
            else:
                _ret_mask = pred_mask[0].cpu().numpy()
                _ret_mask = mask_to_rle(_ret_mask)
                pred_masks.append(_ret_mask)
        else:
            _ret_mask = pred_mask[0].cpu().numpy()
            _ret_mask = mask_to_rle(_ret_mask)
            pred_masks.append(_ret_mask)

        prediction.update({'prediction_masks': pred_masks})
        results.append(prediction)

    tmpdir = './dist_test_temp_res_' + "STBench_" + "_ST"
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        if not os.path.exists("./work_dirs/ST_jsons/"):
            os.mkdir("./work_dirs/ST_jsons/")
        with open(os.path.join("./work_dirs/ST_jsons/", f"STBench_.json"), 'w') as f:
            json.dump(results, f)
        metric = dataset.evaluate(results, './work_dirs')
        print(metric)

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(mask_utils.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

if __name__ == '__main__':
    main()



