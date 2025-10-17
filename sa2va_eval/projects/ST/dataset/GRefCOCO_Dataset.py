import collections
import os
import os.path as osp
import random
from typing import Dict, List
import json
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
import torch
import copy

import mmengine
from mmengine.dataset import BaseDataset
from mmengine import print_log

from mmdet.registry import DATASETS

from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX

from .utils import convert_image_to_patches

NON_VISION_TOKEN = -1
PROMPT_TMPL = '<|im_start|>user\n{input}<|im_end|>\n'


SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

NO_TARGETS_ANSWER_LIST = [
    "No target [SEG]."
]


@DATASETS.register_module()
class GRefCoCoDataset(BaseDataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 split_file: str,
                 special_tokens=None,
                 prompt_template=None,
                 data_prefix=dict(img_path='train2014/'),
                 split: str = 'train',
                 text_mode: str = 'random',
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 patch_size=32,
                 add_cls=False,
                 **kwargs):
        self.split_file = split_file
        self.split = split

        assert text_mode in ['original', 'random', 'concat', 'select_first']
        self.text_mode = text_mode
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            ann_file=ann_file,
            **kwargs,
        )

        self._system = ''

        self.patch_size = patch_size
        self.add_cls = add_cls
        self.tokenizer = BUILDER.build(tokenizer)
        self.tokenizer.vis_beg_tok = "<vision>"
        self.tokenizer.vis_patch_tok = "<vpatch>"
        self.tokenizer.vis_rsep_tok = "<vrow_sep>"
        self.tokenizer.vis_frm_tok = "<vframe_sep>"
        self.tokenizer.vis_end_tok = "</vision>"
        self.tokenizer.vis_cls_tok = "<|vis_cls|>"
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.tokenizer.vis_beg_tok_id = self.tokenizer.convert_tokens_to_ids("<vision>")
        self.tokenizer.vis_patch_tok_id = self.tokenizer.convert_tokens_to_ids("<vpatch>")
        self.tokenizer.vis_rsep_tok_id = self.tokenizer.convert_tokens_to_ids("<vrow_sep>")
        self.tokenizer.vis_frm_tok_id = self.tokenizer.convert_tokens_to_ids("<vframe_sep>")
        self.tokenizer.vis_end_tok_id = self.tokenizer.convert_tokens_to_ids("</vision>")
        self.tokenizer.vis_cls_tok_id = self.tokenizer.convert_tokens_to_ids("<|vis_cls|>")

        self.image_folder = data_root
        self.template = prompt_template
        self.template['INSTRUCTION'] = PROMPT_TMPL
        self.template['SUFFIX'] = '<|endoftext|>'
        self.max_length = max_length

        self.num_classes_per_sample = num_classes_per_sample
        self._max_refetch = 1000

        print("Image GRES dataset, include {} items.".format(len(self)))
    
    @property
    def modality_length(self):
        length_list = []
        for idx in range(len(self)):
            length_list.append(100)
        return length_list
    
    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            if isinstance(object_mask, dict):
                if isinstance(object_mask["counts"], list):
                    # convert to compressed RLE
                    object_mask = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
                m = mask_utils.decode(object_mask)
                m = m.astype(np.uint8).squeeze()
            elif object_mask:
                rles = mask_utils.frPyObjects(object_mask, ori_height, ori_width)
                rle = mask_utils.merge(rles)
                m = mask_utils.decode(rle).astype(np.uint8).squeeze()
            else:
                m = np.zeros((ori_height, ori_width), dtype=np.uint8)
            binary_masks.append(m)
        return binary_masks
    
    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        masks, phrases, no_targets = [], [], []
        instances, text, anno_ids = ann_info['instances'], ann_info['text'], ann_info['anno_ids']
        # index = np.random.choice(range(len(instances)), min(
        #     len(instances), self.num_classes_per_sample))
        index = np.random.choice(range(len(instances)), self.num_classes_per_sample, replace=True)
        for idx in index:
            inst = instances[idx]
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            phrase = phrase.strip()
            phrases.append(phrase)
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            if inst["mask"] is None or inst["mask"][0] is None:
                no_targets.append(True)
                masks.append(binary_mask)
                continue
            assert len(inst["mask"]) == len(anno_ids[idx])

            binary_masks = self.decode_mask(inst["mask"], height, width)
            assert len(binary_masks) == len(inst["mask"])
            for m in binary_masks:
                binary_mask += m
            masks.append(binary_mask)
            no_targets.append(False)

        conversation = []
        for i, phrase in enumerate(phrases):
            question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            if i == 0:
                question = question
            conversation.append({'from': 'human', 'value': question})
            if no_targets[i]:
                conversation.append({'from': 'gpt', 'value': random.choice(NO_TARGETS_ANSWER_LIST)})
            else:
                conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})
        masks = torch.stack([torch.from_numpy(mask) for mask in masks], dim=0)

        ann_info.update({
            'masks': masks,
            'conversations': conversation,
            'image': image_path
        })
        return ann_info
    
    def prepare_image_textual_seq_norowsep(self, h, w):
        image_token_patch_indices = []
        seq = ""
        tok_len = 0

        seq += self.IMG_START_TOKEN
        tok_len += 1
        image_token_patch_indices.append(NON_VISION_TOKEN)

        seq += self.IMG_CONTEXT_TOKEN * (w * h)
        tok_len += (w * h)
        image_token_patch_indices += [idx for idx in range(w * h)]

        seq += self.IMG_END_TOKEN
        tok_len += 1
        image_token_patch_indices.append(NON_VISION_TOKEN)

        if self.add_cls:
            seq += self.CLS_TOKEN
            tok_len += 1
            image_token_patch_indices.append(NON_VISION_TOKEN)
        return seq, tok_len, image_token_patch_indices

    def prepare_data(self, index):
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None
        
        
        # conversation = data_dict['conversations'][:2]
        # mask = data_dict['masks'][0].cpu().numpy()
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # image_path = data_dict['image']
        # image = cv2.imread(image_path)
        # cv2.drawContours(image, contours, -1, color=(255, 255, 0), thickness=2)
        # cv2.imwrite(f"./visualize_gres/{index}.jpg", image)
        # with open(f"./visualize_gres/{index}.json", 'w') as file:
        #     json.dump(conversation, file)

        out_data_dict = {'vision_patch_idx': self.tokenizer.vis_patch_tok_id}
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

            image_patches = convert_image_to_patches(image, self.patch_size)
            # tensor, (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
            h_patches, w_patches = image_patches.shape[:2]
            out_data_dict['vision_patches'] = image_patches.flatten(0, 1).flatten(
                1)  # (n_patches, 3*patch_size*patch_size)
            out_data_dict['patch_nums_per_images'] = (h_patches, w_patches)

            image_token_str, image_token_len, image_token_patch_indices = \
                self.prepare_image_textual_seq_norowsep(
                    image_patches.shape[0], image_patches.shape[1]
                )

            token_dict = self.get_inputid_labels(
                data_dict['conversations'], image_token_str, image_token_patch_indices)
            out_data_dict.update(token_dict)

            out_data_dict.update(token_dict)
        else:
            out_data_dict['patch_nums_per_images'] = (0, 0)
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], "", [])
            out_data_dict.update(token_dict)
        return out_data_dict
    
    def get_inputid_labels(self, conversations, image_token_str, image_token_patch_indices) -> dict:
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
        token_patch_indices = []

        # firstly add the images strs
        image_token_str_tokens = self.tokenizer.encode(image_token_str, add_special_tokens=False)
        input_ids += image_token_str_tokens
        labels += [IGNORE_INDEX] * len(image_token_str_tokens)
        token_patch_indices += image_token_patch_indices

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
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)
            token_patch_indices += [NON_VISION_TOKEN] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)
            token_patch_indices += [NON_VISION_TOKEN] * len(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            token_patch_indices = token_patch_indices[:self.max_length]
            print_log(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}',
                logger='current')
        vision_start_end = self.search_vision_tokens(input_ids)
        return {'input_ids': input_ids, 'labels': labels,
                'vision_patch_indices': token_patch_indices,
                'vision_start_end': vision_start_end,
                }

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data
        
    def search_vision_tokens(self, input_ids):
        image_start_idx = self.tokenizer(self.IMG_START_TOKEN, add_special_tokens=False).input_ids[0]
        image_end_idx = self.tokenizer(self.IMG_END_TOKEN, add_special_tokens=False).input_ids[0]
        if image_start_idx not in input_ids:
            return None
        else:
            start_idx = input_ids.index(image_start_idx)
            end_idx = input_ids.index(image_end_idx)
            return [start_idx + 1, end_idx]

    
    def _join_prefix(self):
        if not mmengine.is_abs(self.split_file) and self.split_file:
            self.split_file = osp.join(self.data_root, self.split_file)

        return super()._join_prefix()
    
    def _init_refs(self):
        """Initialize the refs for GRefCOCO."""
        anns, imgs = {}, {}
        for ann in self.instances['annotations']:
            anns[ann['id']] = ann
        for img in self.instances['images']:
            imgs[img['id']] = img

        anns[-1] = {"segmentation": None, "area": 0.0, "iscrowd": 0, "bbox": None, "category_id": -1, "id": -1}

        refs, ref_to_ann = {}, {}
        for ref in self.splits:
            # ids
            ref_id = ref['ref_id']
            ann_id = ref['ann_id']
            # add mapping related to ref            
            refs[ref_id] = ref
            ref_to_ann[ref_id] = [anns[_ann_id] for _ann_id in ann_id]
            assert len(ref_to_ann[ref_id]) == len(ann_id)

        self.refs = refs
        self.ref_to_ann = ref_to_ann

    def load_data_list(self) -> List[dict]:
        """Load data list.
        Specially, there are no_targets items, where ref['ann_id'] = [-1]
        """
        self.splits = json.load(open(self.split_file, 'rb'))
        self.instances = mmengine.load(self.ann_file, file_format='json')
        self._init_refs()
        img_prefix = self.data_prefix['img_path']

        ref_ids = [
            ref['ref_id'] for ref in self.splits if ref['split'] == self.split
        ]
        image_id_list = []
        for ref_id in ref_ids:
            image_id_list.append(self.refs[ref_id]['image_id'])
        image_annot = {}
        for i in range(len(self.instances['images'])):
            image_annot[self.instances['images'][i]
                        ['id']] = self.instances['images'][i]
        images = []
        for image_id in list(set(image_id_list)):
            images += [image_annot[image_id]]

        grounding_dict = collections.defaultdict(list)
        for ref_id in ref_ids:
            ref = self.refs[ref_id]
            ann_list = [copy.deepcopy(e) for e in self.ref_to_ann[ref_id]]
            ann_list[0].update(ref)
            image_id = ref['image_id']
            grounding_dict[image_id].append(ann_list)



        # full_anno = []
        # for ref_id in ref_ids:
        #     ref = self.refs[ref_id]
        #     ann_list = self.ref_to_ann[ref_id]
        #     for ann in ann_list: ann.update(ref)
        #     full_anno.append(ann_list)
        #     assert len(full_anno[-1]) == len(full_anno[-1][0]['ann_id']), f"num_mask: {len(full_anno[-1])}, num_ids: {len(full_anno[-1][0]['ann_id'])}"

        # image_id_list = []
        # # annotations = []
        # for anno in full_anno:
        #     image_id_list.append(anno[0]['image_id'])
        #     # annotations.append(anno)
        #     # assert len(annotations[-1]) == len(annotations[-1][0]['ann_id']), f"num_mask: {len(annotations[-1])}, num_ids: {len(annotations[-1][0]['ann_id'])}"
        # # annotations = full_anno
        

        # coco_train_id = []
        # image_annot = {}
        # for i in range(len(self.instances['images'])):
        #     coco_train_id.append(self.instances['images'][i]['id'])
        #     image_annot[self.instances['images'][i]
        #                 ['id']] = self.instances['images'][i]

        # images = []
        # for image_id in list(set(image_id_list)):
        #     images += [image_annot[image_id]]
        
        # data_list = []

        # grounding_dict = collections.defaultdict(list)
        # for anno in full_anno:
        #     image_id = int(anno[0]['image_id'])
        #     assert len(anno) == len(anno[0]['ann_id']), f"num_mask: {len(anno)}, num_ids: {len(anno[0]['ann_id'])}"
        #     grounding_dict[image_id].append(anno)

        
        # num_annos = [len(anno) for anno in full_anno]
        # print("num_annos: ", set(num_annos))
        # exit(0)
        
        data_list = []

        join_path = mmengine.fileio.get_file_backend(img_prefix).join_path
        for image in images:
            img_id = image['id']
            instances = []
            sentences = []
            anno_ids = []
            for grounding_anno in grounding_dict[img_id]:
                texts = [x['raw'].lower() for x in grounding_anno[0]['sentences']]
                # random select one text
                if self.text_mode == 'random':
                    idx = random.randint(0, len(texts) - 1)
                    text = [texts[idx]]
                # concat all texts
                elif self.text_mode == 'concat':
                    text = [''.join(texts)]
                # select the first text
                elif self.text_mode == 'select_first':
                    text = [texts[0]]
                # use all texts
                elif self.text_mode == 'original':
                    text = texts
                else:
                    raise ValueError(f'Invalid text mode "{self.text_mode}".')
                ins = [{
                    'mask': [_grounding_anno['segmentation'] for _grounding_anno in grounding_anno],
                    'ignore_flag': 0
                }] * len(text)
                instances.extend(ins)
                sentences.extend(text)
                anno_ids.extend([grounding_anno[0]['ann_id']]*len(text))
            data_info = {
                'img_path': join_path(img_prefix, image['file_name']),
                'img_id': img_id,
                'instances': instances,
                'text': sentences,
                'anno_ids': anno_ids,
            }
            data_list.append(data_info)

        if len(data_list) == 0:
            raise ValueError(f'No sample in split "{self.split}".')

        return data_list
        
