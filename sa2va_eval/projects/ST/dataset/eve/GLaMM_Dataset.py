import copy
import json
import random
import os
import torch
from mmengine import print_log
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from torch.utils.data import Dataset
from ..gcg_process import glamm_openpsg_map_fn, glamm_flickr_map_fn, glamm_granf_map_fn, glamm_refcocog_map_fn
from projects.ST.eve.mm_utils import process_images
from projects.ST.eve.constants import IMAGE_TOKEN_INDEX

class GlammSegmDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_preprocessor,
                 data_path=None,
                 special_tokens=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 patch_size=32,
                 repeats=1,
                 **kwargs):

        self.repeats = repeats
        self._system = ''

        self.patch_size = patch_size
        self.tokenizer = BUILDER.build(tokenizer)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        self.image_preprocessor = BUILDER.build(image_preprocessor)

        self.image_folder = image_folder
        json_data = self.json_file_preprocess(data_path)
        self.datas = json_data

        self.template = prompt_template
        self.max_length = max_length

        self._max_refetch = 1000

        print("GLaMM dataset, include {} items.".format(len(self.datas)))

    def dataset_map_fn(self, data_dict):
        data_dict = glamm_refcocog_map_fn(data_dict)
        return data_dict

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.datas:
            cur_len = 100
            length_list.append(cur_len)
        return length_list * self.repeats

    def __len__(self):
        return len(self.datas) * self.repeats

    def real_len(self):
        return len(self.datas)

    def json_file_preprocess(self, data_path):
        with open(data_path, 'r') as f:
            json_data = json.load(f)
        return json_data

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in object_mask:
                rles = mask_utils.frPyObjects([seg], ori_height, ori_width)
                m = mask_utils.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

    def _parse_annotations(self, ann_info):

        result = self.dataset_map_fn(ann_info)
        ann_info.update(result)

        image_path = ann_info['image']
        image_path = os.path.join(self.image_folder, image_path)
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        masks = self.decode_mask(ann_info['masks'], ori_height=height, ori_width=width)
        if masks is None:
            return None
        ann_info.update({
            'masks': masks,
            'conversations': ann_info["conversation"],
            'image': image_path,
        })
        return ann_info

    def prepare_data(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.datas[index])
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

class GlammRefcocogDataset(GlammSegmDataset):
    def json_file_preprocess(self, data_path):
        json_data = json.load(open(data_path))

        # convert {id: dict} to dict(..., id=xx)
        for idx in range(len(json_data)):
            id = list(json_data[idx].keys())[0]
            json_data[idx] = json_data[idx][id]
            json_data[idx].update({'id': id})
        return json_data

class GlammGrandfDataset(GlammSegmDataset):
    def dataset_map_fn(self, data_dict):
        data_dict = glamm_granf_map_fn(data_dict)
        return data_dict

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)

            for rle in object_mask:
                m = mask_utils.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

class GlammOpenpsgDataset(GlammGrandfDataset):
    def dataset_map_fn(self, data_dict):
        data_dict = glamm_openpsg_map_fn(data_dict)
        return data_dict

class GlammFlickerDataset(GlammSegmDataset):
    def dataset_map_fn(self, data_dict):
        data_dict = glamm_flickr_map_fn(data_dict)
        return data_dict

    def json_file_preprocess(self, data_path):
        def filter_images(data_infos, min_size):
            return [i for i, info in enumerate(data_infos) if min(info['width'], info['height']) >= min_size]

        # convert {id: dict} to dict(..., id=xx)
        from pycocotools.coco import COCO
        self.coco = COCO(data_path)
        self.image_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        removed_img_count = 0
        for img_id in self.image_ids:
            info = self.coco.loadImgs([img_id])[0]
            if len(info['caption'].split(' ')) < 3:
                removed_img_count += 1
                continue
            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Non-unique annotation IDs in '{data_path}'!"
        print(f'Removed {removed_img_count} images.')
        data_infos = [data_infos[i] for i in filter_images(data_infos, min_size=32)]

        # obtain_annotations
        for data_info in data_infos:
            ann_ids = self.coco.getAnnIds(imgIds=data_info['id'])
            ann_info = self.coco.loadAnns(ann_ids)
            data_info.update({'ann_info': ann_info})
        return data_infos

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = mask_utils.decode(object_mask).astype(np.uint8)
            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks