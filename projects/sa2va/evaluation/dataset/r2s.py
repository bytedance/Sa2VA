import os
import numpy as np
from PIL import Image
from pycocotools import mask as _mask
from projects.llava_sam2.evaluation.utils import  master_only
import json


def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = _mask.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask


THINK_TEMPLATE = '<image>' + "\n" + "{sent}\n\nYou should first think about the reasoning process in the mind and then provides the user with the answer. Please respond with segmentation mask in both the thinking process and the answer."
TEMPLATE_TEMPLATE = 'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answers here </answer>.'

REGULAR_TEMPLATE = '<image>' + "\n" + "{sent}\n\nPlease respond with segmentation mask in the answer."


class R2SDataset:
    METAINFO: dict = dict(name='R2SDataset')

    def __init__(self,
                 image_folder,
                 dataset_name,
                 json_path=None,
                 split='val',
                 reasoning=False,
                 ):
        self.split = split
        self._set_attribute(dataset_name)
        assert json_path is not None, "json_path should be provided"
        assert os.path.exists(json_path), f"json_path {json_path} does not exist"
        
        json_datas = json.load(open(json_path))   
        self.json_datas = json_datas
        self.image_folder = image_folder
        
        self.reasoning = reasoning


    def _set_attribute(self, dataset_name):
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.json_datas)

    def __getitem__(self, index):
        data_dict = self.json_datas[index]

        raw_question = data_dict['question']
        if self.reasoning:
            question = THINK_TEMPLATE.format(sent=raw_question)
            question = "{question}\n\n{template}".format(question=question, template=TEMPLATE_TEMPLATE)
        else:
            question = REGULAR_TEMPLATE.format(raw_question)
            
        image_file = data_dict['image_name']
        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        data_dict['image'] = image
        data_dict['text'] = question
        data_dict['img_id'] = str(index)
        data_dict['image_path'] = image_file

        del data_dict['question'], data_dict['image_name']
        return data_dict

    @master_only
    def evaluate(self, result):
        raise NotImplementedError("Please implement the evaluate function for your dataset.")


if __name__ == "__main__":
    data = R2SDataset(
        image_folder='data/dense/images',
        dataset_name='R2S',
        json_path='data/dense/eval_ann_107.json',
        split='val',
        reasoning=True,
    )

    for item in data:
        print(item['text'])
        print(item['image'].size)
        print(item['img_id'])
        print(item['image_path'])
        print(item['reasoning'])
        print(item['answers'])
