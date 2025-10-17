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
from .utils import convert_image_to_patches
from torch.utils.data import Dataset

NON_VISION_TOKEN = -1
PROMPT_TMPL = '<|im_start|>user\n{input}<|im_end|>\n'

class MuseDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"

    def __init__(self,
                 train2014_image_folder,
                 train2017_image_folder,
                 val2017_image_folder,
                 json_file,
                 special_tokens=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 patch_size=32,
                 add_cls=False,
                 **kwargs):
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

        self.train2014_image_folder = train2014_image_folder
        self.val2017_image_folder = val2017_image_folder
        self.train2017_image_folder = train2017_image_folder

        self.json_file = json_file

        with open(json_file, 'r') as f:
            self.datas = json.load(f)

        self.template = prompt_template
        self.template['INSTRUCTION'] = PROMPT_TMPL
        self.template['SUFFIX'] = '<|endoftext|>'
        self.max_length = max_length

        self.num_classes_per_sample = num_classes_per_sample
        self._max_refetch = 1000

        print("COCO panoptic dataset, include {} items.".format(len(self.datas)))

    @property
    def modality_length(self):
        length_list = []
        for idx in range(len(self.datas)):
            length_list.append(100)
        return length_list

    def __len__(self):
        return len(self.datas)

    def _parse_annotations(self, ann_info):

        if 'file_name' in ann_info:
            image_path = os.path.join(self.train2014_image_folder, ann_info['file_name'])
        else:
            if 'train2017' in ann_info['coco_url']:
                image_path = os.path.join(self.train2017_image_folder, ann_info['coco_url'].split('/')[-1])
            else:
                image_path = os.path.join(self.val2017_image_folder, ann_info['coco_url'].split('/')[-1])

        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        anns = ann_info['ann_list']
        question = ann_info['questions'] if 'questions' in ann_info else None
        gt_answer = ann_info['answers'] if 'answers' in ann_info else None
        if question is not None:
            text_answers = ann_info['text_answers'] if 'text_answers' in ann_info else [None] * len(gt_answer)
        else:
            text_answers = None

        if len(anns) == 0:
            return None

        category_ids = [ann['category_id'] for ann in anns]
        category_ids = list(set(category_ids))
        sampled_num = min(self.num_classes_per_sample, len(category_ids))
        sampled_category_ids = np.random.choice(category_ids, size=sampled_num, replace=False)

        sampled_sents = question
        sampled_answers = gt_answer
        masks = []
        sampled_masks = masks
        sample_text_answers = text_answers

        image_name = image_path.split("/")[-1]
        questions = []
        answers = []
        use_assign_list = []

        if question is not None:
            for text, answer_list, text_answer in zip(sampled_sents, sampled_answers, sample_text_answers):
                # if is_sentence:
                questions.append(text + " Please respond with segmentation mask.")
                for answer in answer_list:
                    rle = mask_utils.frPyObjects(answer["segmentation"], ann_info["height"], ann_info["width"])
                    m = mask_utils.decode(rle)
                    if len(m.shape) > 2:
                        # assert m.shape[-1] == 1, m.shape
                        m = np.sum(m, axis=2)  # so
                    m = m.astype(np.uint8)
                    masks.append(m)

                if text_answer is not None:
                    if text_answer.count('{seg}') != len(answer_list):
                        return None
                    # _text_answer = text_answer.format(seg='[SEG]')
                    _text_answer = text_answer.replace('{seg}', '[SEG]')
                    answers.append(_text_answer)
                    use_assign_list.append(False)
                else:
                    target_list = [
                        a['rephrased_name'] if (random.random() > 0.1 and 'rephrased_name' in a) else a['category_name']
                        for a in answer_list]
                    target_answer = []
                    separate_answer = random.randint(0, 1)
                    _seg = ['[SEG]'] * len(target_list)
                    if len(target_list) > 1:
                        part1 = ', '.join(_seg[:-1])
                        part2 = ' and ' + _seg[-1]
                        _seg = part1 + part2
                    else:
                        _seg = _seg[0]

                    if separate_answer:
                        choice_list = [
                            "{class_name} is [SEG].",
                            "The segmentation result of {class_name} is [SEG].",
                            "[SEG]."
                        ]
                        answer_temp = random.choice(choice_list)
                        use_assign = False if "{class_name}" in answer_temp else True
                        for i, sampled_cls in enumerate(target_list):
                            _answer_temp = answer_temp.format(
                                class_name=sampled_cls) if "{class_name}" in answer_temp else answer_temp
                            target_answer.append(_answer_temp[:-1])
                        if len(target_answer) > 1:
                            part1 = ', '.join(target_answer[:-1])
                            part2 = ' and ' + target_answer[-1]
                            target_answer = part1 + part2 + '.'
                        else:
                            target_answer = target_answer[0] + '.'
                    else:
                        answer_temp = random.choice([
                            "{class_name} are {seg}, separately.",
                            "{class_name} are {seg}.",
                            "Sure, {class_name} are {seg}, separately.",
                            "Sure, {class_name} are {seg}.",
                            "the segmentation result of {class_name} are {seg}.",
                            "the segmentation result of {class_name} are {seg}, separately.",
                            "Sure, the segmentation result of {class_name} are {seg}.",
                            "Sure, the segmentation result of {class_name} are {seg}, separately.",
                            "Sure, they are {seg}.",
                            "They are {seg}.",
                            "{seg}."
                        ])
                        _answer_temp = answer_temp.format(class_name=', '.join(target_list).lower(),
                                                          seg=_seg) if "{class_name}" in answer_temp else answer_temp.format(
                            seg=_seg)
                        use_assign = False if "{class_name}" in answer_temp else True
                        _answer_temp = _answer_temp
                        target_answer = _answer_temp

                    answers.append(target_answer)
                    use_assign_list.append(use_assign)
        else:
            return None

        conversation = []
        i = 0
        while i < len(questions):
            conversation.append({'from': 'human', 'value': questions[i]})
            conversation.append({'from': 'gpt', 'value': answers[i]})
            i += 1


        ret_masks = masks
        if len(ret_masks) == 0:
            return None
        masks = torch.stack([torch.from_numpy(mask) for mask in ret_masks], dim=0)

        # print(conversation)
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
        data_dict = self.datas[index]
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

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

    def search_vision_tokens(self, input_ids):
        image_start_idx = self.tokenizer(self.IMG_START_TOKEN, add_special_tokens=False).input_ids[0]
        image_end_idx = self.tokenizer(self.IMG_END_TOKEN, add_special_tokens=False).input_ids[0]
        if image_start_idx not in input_ids:
            return None
        else:
            start_idx = input_ids.index(image_start_idx)
            end_idx = input_ids.index(image_end_idx)
            return [start_idx + 1, end_idx]
