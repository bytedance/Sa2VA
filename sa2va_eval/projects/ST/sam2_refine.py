import json
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import numpy as np
from PIL import Image
import argparse
from pycocotools import mask as _mask
import tqdm

sam_prefix = '/mnt/bn/xiangtai-training-data-video/dataset/segmentation_datasets/sam_v_full/sav_000/sav_train/sav_000/'
coco_prefix = '/mnt/bn/zhnagtao-lq/xiangtai-mnt/tt_vlm/data/glamm_data/images/coco2014/train2014/'
sam_p2 = 'data/sa_eval/'

checkpoint = "./checkpoints/sam2_hiera_large.pt"
model_cfg = "configs/sam2/sam2_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def parse_args():
    parser = argparse.ArgumentParser(description='refine the res masks with sam2')
    parser.add_argument('--json-file', help='the pred json file path.')
    parser.add_argument('--save-path', help='the save json path')
    args = parser.parse_args()
    return args

def get_bbox_from_mask(mask):
    """
    从二进制掩码中提取边界框坐标

    参数:
        mask: numpy array, 形状为 (H, W) 的二进制掩码，前景为1，背景为0

    返回:
        tuple: (x_min, y_min, x_max, y_max)
        如果掩码中没有前景像素，返回None
    """
    # 确保输入是二维数组
    if len(mask.shape) != 2:
        raise ValueError("输入掩码必须是二维数组")

    # 找到所有非零像素的坐标
    y_indices, x_indices = np.nonzero(mask)

    # 如果没有前景像素，返回None
    if len(x_indices) == 0 or len(y_indices) == 0:
        return None

    # 计算边界框坐标
    x_min = np.min(x_indices)
    x_max = np.max(x_indices)
    y_min = np.min(y_indices)
    y_max = np.max(y_indices)

    return (x_min, y_min, x_max, y_max)

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle

def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = _mask.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask

args = parse_args()
# read the json
with open(args.json_file, 'r') as f:
    pred_results = json.load(f)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    for pred_result in tqdm.tqdm(pred_results):
        image_path = pred_result['image_path']
        rles = pred_result['prediction_masks']
        # print("len_rles: ", len(rles))
        image = Image.open(image_path).convert('RGB')
        predictor.set_image(image)

        refined_rles = []

        for rle in rles:
            pred_mask = rle_to_mask(rle)[0, :, :]
            # print(pred_mask.shape)
            box = get_bbox_from_mask(pred_mask)
            if box is not None:
                box = list(box)
                # print(box)
                pre_refine_box = box
                masks, ious_np, _ = predictor.predict(box=np.array(box)[None, :], multimask_output = False,)
                max_score = np.max(ious_np)
                masks = masks[ious_np == max_score]
                # print(masks.shape)
                post_refine_box = get_bbox_from_mask(masks[0])
                print("pre_bbox: ", pre_refine_box, "  ", "post_bbox: ", post_refine_box)
                rle = mask_to_rle(masks)
            refined_rles.append(rle)

        pred_result.update({'prediction_masks': refined_rles})

with open(args.save_path, 'w') as f:
    json.dump(pred_results, f)

