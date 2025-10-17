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
        question_masks = np.stack(question_masks, axis=0)
        # gt_masks = torch.from_numpy(gt_masks)

        return {
            "image": image, "gt_masks": gt_masks,
            "question": question, "img_id": int(idx), "image_path": image_path,
            "prompt_masks": question_masks, "prompt_ids": input_parompt_ids,
        }

def main():
    image_folder = "/mnt/bn/zhnagtao-lq/xiangtai-mnt/internvl_yk/vlm_before_20241125/data/masa_sam_500k/"
    save_path = "./stbench_vres/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    dataset = STBench(
        image_folder=image_folder,
        json_file="./STBench_VRES.json"
    )

    for i in tqdm.tqdm(range(len(dataset))):
        data_item = dataset[i]
        question = data_item["question"]
        prompt_masks = data_item["prompt_masks"]
        answer_masks = data_item["gt_masks"]
        image = data_item["image"]
        question_image = show_mask_pred(copy.deepcopy(image), prompt_masks)
        answer_image = show_mask_pred(copy.deepcopy(image), answer_masks)

        _folder_path = os.path.join(save_path, f"{i}")
        if not os.path.exists(_folder_path):
            os.mkdir(_folder_path)
        question_image.save(os.path.join(_folder_path, "question.png"))
        answer_image.save(os.path.join(_folder_path, "answer.png"))
        with open(os.path.join(_folder_path, "text.txt"), "w") as f:
            f.write(question)

def show_mask_pred(image, masks):
    from PIL import Image
    import numpy as np

    colors = [
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255),
              (255, 0, 0), (0, 255, 0), (0, 0, 255),
    ]

    _color_prefix = random.randint(0, len(colors) - 1)

    # masks = np.stack(masks, dim=0)
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



