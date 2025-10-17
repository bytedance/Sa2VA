import copy
import random
import os
from mmengine import print_log
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from mmdet.datasets.refcoco import RefCocoDataset
from .utils import convert_image_to_patches, convert_mask_to_patches
from ..configs.vp_data_ablation.vp_ablation_osprey_pixel2cap_vp_tokens import vision_patch_size

DETAILED_QUESTIONS =  [
    'Please give me a short description of <region>.',
]

NON_VISION_TOKEN = -1
PROMPT_TMPL = '<|im_start|>user\n{input}<|im_end|>\n'

class ReferVisualPromptDataset(RefCocoDataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"

    def __init__(self,
                 data_root,
                 ann_file=None,
                 split_file=None,
                 special_tokens=None,
                 prompt_template=None,
                 data_prefix=dict(img_path='train2014/'),
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 patch_size=32,
                 add_cls=False,
                 prompt_numbers=10,
                 **kwargs):
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=None,
            ann_file=ann_file,
            split_file=split_file,
            **kwargs,
        )

        self.prompt_numbers = prompt_numbers

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

        print("Refcoco/+/g visual prompt dataset, include {} items.".format(len(self)))

    @property
    def modality_length(self):
        length_list = []
        for idx in range(len(self)):
            length_list.append(100)
        return length_list

    def sort_masks_by_area(self, masks):
        areas = []
        for mask in masks:
            area = np.sum(mask)
            areas.append(area)
        indexes = np.argsort(np.array(areas))[::-1]  # sort the mask from large area to small area
        return indexes

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        masks, phrases = [], []
        instances, text = ann_info['instances'], ann_info['text']

        for idx in range(len(instances)):
            inst = instances[idx]
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            phrases.append(phrase)
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in inst["mask"]:
                rles = mask_utils.frPyObjects([seg], height, width)
                m = mask_utils.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()
            masks.append(binary_mask)

        descriptions = phrases
        if len(masks) == 0:
            return None
        indexes = list(range(0, len(descriptions)))
        random.shuffle(indexes)

        _num_classes_per_sample = random.randint(1, self.num_classes_per_sample)
        masks = [masks[idx] for idx in indexes][:_num_classes_per_sample]
        descriptions = [descriptions[idx] for idx in indexes][:_num_classes_per_sample]

        prompt_indexes = [i_p for i_p in range(self.prompt_numbers)]
        random.shuffle(prompt_indexes)
        selected_prompt_indexes = prompt_indexes[:len(masks)]
        selected_prompt_tokens = [f"<Prompt{i_p}>" for i_p in selected_prompt_indexes]

        # for none prompt
        none_prompt = True
        not_selected_prompt_indexes = prompt_indexes[len(masks):]
        not_selected_prompt_tokens = [f"<Prompt{i_p}>" for i_p in not_selected_prompt_indexes]

        conversation = []
        ret_masks = []
        # sorted_prompt_indexes = copy.deepcopy(selected_prompt_indexes)
        # sorted_prompt_indexes.sort()
        # conversation.append({'from': 'human', 'value': "What's the prompts in the image?"})
        # answer = ""
        # for _idx in sorted_prompt_indexes:
        #     answer += f"<Prompt{_idx}>"
        #     answer += " "
        # answer = answer.strip()
        # conversation.append({'from': 'gpt', 'value': answer})
        for i, (mask, obj_description) in enumerate(zip(masks, descriptions)):
            if none_prompt and random.random() < 0.05:
                question = random.choice(DETAILED_QUESTIONS).replace("<region>", not_selected_prompt_tokens[0])
                conversation.append({'from': 'human', 'value': question})
                conversation.append({'from': 'gpt', 'value': f"{not_selected_prompt_tokens[0]} is not in the image."})
                none_prompt = False
            question = random.choice(DETAILED_QUESTIONS).replace("<region>", selected_prompt_tokens[i])
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': f"{selected_prompt_tokens[i]}: {obj_description}"})
            ret_masks.append(mask)
        # print(conversation)
        if len(ret_masks) == 0:
            return None

        # sort the mask according to the area
        indexes = self.sort_masks_by_area(ret_masks)
        ret_masks = [ret_masks[idx] for idx in indexes]
        selected_prompt_tokens = [selected_prompt_tokens[idx] for idx in indexes]
        selected_prompt_indexes = [selected_prompt_indexes[idx] for idx in indexes]

        masks = ret_masks

        ann_info.update({
            'masks': masks,
            'prompt_tokens': selected_prompt_indexes,
            'conversations': conversation,
            'image': image_path
        })
        # print(conversation)
        # self.visualize(image_path, masks, conversation, selected_prompt_tokens)
        return ann_info

    def prepare_image_textual_seq_norowsep_with_vp(self, h, w, masks=None, vp_tokens=None):
        image_token_patch_indices = []
        vp_token_indices = []
        seq = ""
        tok_len = 0

        seq += self.IMG_START_TOKEN
        tok_len += 1
        image_token_patch_indices.append(NON_VISION_TOKEN)
        vp_token_indices.append(NON_VISION_TOKEN)

        seq += self.IMG_CONTEXT_TOKEN * (w * h)
        tok_len += (w * h)
        image_token_patch_indices += [idx for idx in range(w * h)]

        _vision_vp_tokens = np.zeros((w * h), dtype=np.int64) - 2  # -2 for none vp index
        if masks is not None:
            assert len(masks) == len(vp_tokens)
            for mask, vp_index in zip(masks, vp_tokens):
                mask = mask.flatten(0, 1).bool().numpy()
                _vision_vp_tokens[mask] = vp_index
        _vision_vp_tokens = _vision_vp_tokens.tolist()
        vp_token_indices += _vision_vp_tokens

        seq += self.IMG_END_TOKEN
        tok_len += 1
        image_token_patch_indices.append(NON_VISION_TOKEN)
        vp_token_indices.append(NON_VISION_TOKEN)

        if self.add_cls:
            seq += self.CLS_TOKEN
            tok_len += 1
            image_token_patch_indices.append(NON_VISION_TOKEN)
            vp_token_indices.append(NON_VISION_TOKEN)
        return seq, tok_len, image_token_patch_indices, vp_token_indices

    def prepare_data(self, index):
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {'vision_patch_idx': self.tokenizer.vis_patch_tok_id}

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
            masks_prompt_patches = [convert_mask_to_patches(mask, self.patch_size) for mask in data_dict["masks"]]
            # tensor, (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
            h_patches, w_patches = image_patches.shape[:2]
            out_data_dict['vision_patches'] = image_patches.flatten(0, 1).flatten(
                1)  # (n_patches, 3*patch_size*patch_size)
            out_data_dict['patch_nums_per_images'] = (h_patches, w_patches)

            image_token_str, image_token_len, image_token_patch_indices, vp_token_indices = \
                self.prepare_image_textual_seq_norowsep_with_vp(
                    image_patches.shape[0], image_patches.shape[1],
                    masks=masks_prompt_patches, vp_tokens=data_dict["prompt_tokens"],
                )

            token_dict = self.get_inputid_labels(
                data_dict['conversations'], image_token_str, image_token_patch_indices, vp_token_indices)
            # out_data_dict.update(token_dict)
            out_data_dict.update(token_dict)
        else:
            out_data_dict['patch_nums_per_images'] = (0, 0)
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], "", [], [])
            out_data_dict.update(token_dict)
        return out_data_dict

    def get_inputid_labels(self, conversations, image_token_str, image_token_patch_indices, vp_token_indices) -> dict:
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
        vp_patch_indices = []

        # firstly add the images strs
        image_token_str_tokens = self.tokenizer.encode(image_token_str, add_special_tokens=False)
        input_ids += image_token_str_tokens
        labels += [IGNORE_INDEX] * len(image_token_str_tokens)
        token_patch_indices += image_token_patch_indices
        vp_patch_indices += vp_token_indices

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
            vp_patch_indices += [NON_VISION_TOKEN] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)
            token_patch_indices += [NON_VISION_TOKEN] * len(output_encode)
            vp_patch_indices += [NON_VISION_TOKEN] * len(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            token_patch_indices = token_patch_indices[:self.max_length]
            vp_patch_indices = vp_patch_indices[:self.max_length]
            print_log(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}',
                logger='current')
        vision_start_end = self.search_vision_tokens(input_ids)
        # print(self.tokenizer.decode(
        #     input_ids, skip_special_tokens=False).strip())
        return {'input_ids': input_ids, 'labels': labels,
                'vision_patch_indices': token_patch_indices,
                'vision_start_end': vision_start_end,
                'vp_token_indices': vp_patch_indices,
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
