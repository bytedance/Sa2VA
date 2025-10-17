import copy
import json
import random
import os
import torch
from mmengine import print_log
from PIL import Image
import numpy as np
from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from torch.utils.data import Dataset
from panopticapi.utils import rgb2id
from projects.ST.eve.mm_utils import process_images
from projects.ST.eve.constants import IMAGE_TOKEN_INDEX

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]



SEG_QUESTIONS = [
    "Can you segment the object in this image according to the given object description? The object description: {class_name}",
    "Please segment the object in this image according to the given object description. The object description: {class_name}",
    "Could you provide a segmentation mask for the object in this image according to the given object description? The object description: {class_name}",
    "Please identify and segment the object in this image according to the given object description. The object description: {class_name}",
]

class PixelCapSegDataset(Dataset):
    def __init__(self,
                 image_folder,
                 pano_rgb_folder,
                 pano_json,
                 image_preprocessor,
                 special_tokens=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 patch_size=32,
                 **kwargs):
        self._system = ''

        self.patch_size = patch_size
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.image_preprocessor = BUILDER.build(image_preprocessor)

        self.image_folder = image_folder
        self.pano_rgb_folder = pano_rgb_folder
        self.pano_json = pano_json

        self.datas = self.read_pano_json()

        self.template = prompt_template
        self.max_length = max_length

        self.num_classes_per_sample = num_classes_per_sample
        self._max_refetch = 1000

        print("Pixel2Cap segm dataset, include {} items.".format(len(self.datas)))

    @property
    def modality_length(self):
        length_list = []
        for idx in range(len(self.datas)):
            length_list.append(100)
        return length_list

    def __len__(self):
        return len(self.datas)

    def read_pano_json(self):
        with open(self.pano_json, 'r') as f:
            json_info = json.load(f)
        ret = []
        for ann in json_info["annotations"]:
            image_id = int(ann["image_id"])
            image_file = os.path.join(self.image_folder, os.path.splitext(ann["file_name"])[0] + ".jpg")
            label_file = os.path.join(self.pano_rgb_folder, ann["file_name"])
            segments_info = copy.deepcopy(ann["segments_info"])
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "pan_seg_file_name": label_file,
                    "segments_info": segments_info,
                }
            )
        return ret

    def sort_masks(self, masks):
        ret = []
        sx = []
        for mask in masks:
            _y,_x = np.nonzero(mask)
            sx.append(np.min(_x))
        indexes = np.argsort(np.array(sx))
        for index in indexes:
            ret.append(masks[index])
        return ret

    def _parse_annotations(self, ann_info):
        image_path = ann_info['file_name']
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        pano_seg = Image.open(ann_info["pan_seg_file_name"]).convert('RGB')
        segment_infos = ann_info["segments_info"]
        pan_seg_gt = rgb2id(np.array(pano_seg))

        masks, descriptions = [], []
        for segment_info in segment_infos:
            if len(segment_info["description"]) > 5:
                masks.append(pan_seg_gt == segment_info["id"])
                descriptions.append(segment_info["description"])

        if len(masks) == 0:
            return None
        indexes = list(range(0, len(descriptions)))
        random.shuffle(indexes)

        masks = [masks[idx] for idx in indexes][:self.num_classes_per_sample]
        descriptions = [descriptions[idx] for idx in indexes][:self.num_classes_per_sample]

        conversation = []
        ret_masks = []
        for mask, obj_description in zip(masks, descriptions):
            question = random.choice(SEG_QUESTIONS).format(class_name=obj_description)
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})
            ret_masks.append(mask)
        # print(conversation)
        if len(ret_masks) == 0:
            return None
        masks = torch.stack([torch.from_numpy(mask) for mask in ret_masks], dim=0)

        ann_info.update({
            'masks': masks,
            'conversations': conversation,
            'image': image_path
        })
        return ann_info

    def get_inputid_labels(self, conversations) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]

        # remove image token from text conversation
        for i, msg in enumerate(conversations):
            if msg['from'] == 'human':
                # change to 1 image
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
                if i == 0:
                    msg['value'] = "<image>\n" + msg['value']
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []

        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text

                if "<image>" in input_text:
                    input_encode = []
                    text_parts = input_text.split("<image>")
                    for i_part, _text_part in enumerate(text_parts):
                        input_encode += self.tokenizer.encode(_text_part, add_special_tokens=False)
                        input_encode += [IMAGE_TOKEN_INDEX]
                    input_encode = input_encode[:-1]
                else:
                    input_encode = self.tokenizer.encode(input_text, add_special_tokens=False)
            else:
                input_encode = self.tokenizer.encode(input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            print_log(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}',
                logger='current')
        return {'input_ids': input_ids, 'labels': labels, }

    def prepare_data(self, index):
        data_dict = self.datas[index]
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {}
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            try:
                image = Image.open(image_file).convert('RGB')
                out_data_dict['image'] = image
            except Exception as e:
                print(f'Error: {e}', flush=True)
                print_log(f'Error: {e}', logger='current')
                return None

            # process the image
            image = process_images([image], self.image_preprocessor)[0]
            h, w = image.shape[-2:]
            out_data_dict["pixel_values"] = image

            # init the void visual prompt embeddings
            out_data_dict['patch_nums_per_images'] = ((h // self.patch_size), (w // self.patch_size + 1))
            vision_patch_nums = (h // self.patch_size) * (w // self.patch_size + 1) + 1

            visual_prompt_indexes = [-1] * vision_patch_nums
            out_data_dict['visual_prompt_indexes'] = visual_prompt_indexes

            token_dict = self.get_inputid_labels(data_dict['conversations'])
            out_data_dict.update(token_dict)
        else:
            out_data_dict['patch_nums_per_images'] = (0, 0)
            out_data_dict['visual_prompt_indexes'] = []
            token_dict = self.get_inputid_labels(
                data_dict['conversations'])
            out_data_dict.update(token_dict)
        assert -200 in out_data_dict["input_ids"]
        return out_data_dict

    def _rand_another(self):
        idx = random.randint(0, len(self.datas))
        return idx

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data
