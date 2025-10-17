import json
import random

import torch

from PIL import Image
from pycocotools import mask as mask_utils
import copy
import os
import numpy as np
import tqdm

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
        object_annotations = json_data["object_annotations"]

        captions = []
        masks = []
        for object_annotation in object_annotations:
            mask = mask_utils.decode(object_annotation["segmentation"])
            caption = object_annotation["caption"]
            captions.append(caption)
            masks.append(mask)

        image = Image.open(image_path).convert('RGB')

        return {
            "image": image, "masks": masks, "captions": captions
        }

def main():
    image_folder = "/mnt/bn/zhnagtao-lq/xiangtai-mnt/internvl_yk/vlm_before_20241125/data/masa_sam_500k/"
    save_path = "./stbench_detail_cap/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dataset = STBench(
        image_folder=image_folder,
        json_file="./STBench_Detailed_Caption.json"
    )

    _i_sample = 0
    for i in tqdm.tqdm(range(len(dataset))):
        data_item = dataset[i]
        object_masks = data_item["masks"]
        captions = data_item["captions"]
        image = data_item["image"]

        for object_mask, caption in zip(object_masks, captions):
            question_image = show_mask_pred(copy.deepcopy(image), [object_mask])
            _folder_path = os.path.join(save_path, f"{_i_sample}")
            _i_sample += 1
            if not os.path.exists(_folder_path):
                os.mkdir(_folder_path)
            question_image.save(os.path.join(_folder_path, "question.png"))
            with open(os.path.join(_folder_path, "text.txt"), "w") as f:
                f.write(caption)

def show_mask_pred(image, masks):
    from PIL import Image
    import numpy as np

    colors = [
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255),
              (255, 0, 0), (0, 255, 0), (0, 0, 255),
    ]

    _color_prefix = random.randint(0, len(colors) - 1)

    masks = np.stack(masks, axis=0)
    _mask_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    # print(masks.shape)
    for i, mask in enumerate(masks):
        color = colors[(i + _color_prefix) % len(colors)]
        _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
        _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
        _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]

    image = np.array(image)
    image = image * 0.5 + _mask_image * 0.5
    image = image.astype(np.uint8)
    image = Image.fromarray(image)

    return image

if __name__ == '__main__':
    main()



