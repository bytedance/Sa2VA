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
from .utils import convert_image_to_patches, convert_mask_to_patches
from torch.utils.data import Dataset
from panopticapi.utils import rgb2id
from .utils import has_multiple_connected_components, maskPrompt2boxPrompt, maskPrompt2pointPrompt

DETAILED_QUESTIONS =  [
    'Can you provide me with a detailed description of the region in the picture marked by <region>?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail?",
    'What can you tell me about the region indicated by <region> in the image?',
    "I'd like to know more about the area in the photo labeled <region>. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail?',
    'What details can you give me about the region outlined by <region> in the photo?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image.',
    'Can you give me a detailed account of the region labeled as <region> in the picture?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail?",
    'What is the region outlined by <region> in the picture like? Could you give me a detailed description?',
    'Can you provide me with a detailed description of the region in the picture marked by <region>, please?',
    "I'm curious about the region represented by <region> in the picture. Could you describe it in detail, please?",
    'What can you tell me about the region indicated by <region> in the image, exactly?',
    "I'd like to know more about the area in the photo labeled <region>, please. Can you give me a detailed description?",
    'Could you describe the region shown as <region> in the picture in great detail, please?',
    'What details can you give me about the region outlined by <region> in the photo, please?',
    'Please provide me with a comprehensive description of the region marked with <region> in the image, please.',
    'Can you give me a detailed account of the region labeled as <region> in the picture, please?',
    "I'm interested in learning more about the region represented by <region> in the photo. Can you describe it in detail, please?",
    'What is the region outlined by <region> in the picture like, please? Could you give me a detailed description?',
    'Please describe the region <region> in the image in detail.',
    'Can you offer a thorough analysis of the region <region> in the image?',
    'Could you elaborate on the region highlighted by <region> in the picture provided?',
    'Please share more information about the zone emphasized with <region> in the photo.',
    'What insights can you give about the area denoted by <region> in the image presented?',
    'Can you share a comprehensive rundown of the region denoted by <region> in the presented image?',
    "I'd like to know more about the region highlighted by <region> in the picture provided.",
    'Work through the important details of the area <region> in the image.',
    'Illustrate the area represented by <region> through a descriptive explanation.',
    'Examine the region <region> closely and share its details.'
]

NON_VISION_TOKEN = -1
PROMPT_TMPL = '<|im_start|>user\n{input}<|im_end|>\n'

class PixelCapVisualPromptDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"

    def __init__(self,
                 image_folder,
                 pano_rgb_folder,
                 pano_json,
                 special_tokens=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 patch_size=32,
                 add_cls=False,
                 prompt_numbers=10,
                 repeats=1,
                 **kwargs):
        self._system = ''
        self.prompt_numbers = prompt_numbers
        self.repeats = repeats

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

        self.image_folder = image_folder
        self.pano_rgb_folder = pano_rgb_folder
        self.pano_json = pano_json

        self.datas = self.read_pano_json()

        self.template = prompt_template
        self.template['INSTRUCTION'] = PROMPT_TMPL
        self.template['SUFFIX'] = '<|endoftext|>'
        self.max_length = max_length

        self.num_classes_per_sample = num_classes_per_sample
        self._max_refetch = 1000

        print("Pixel2Cap visual prompt dataset, include {} items.".format(len(self.datas)))

        self.vis_idx = 0
        self.vis_folder = "./data_vis/"
        if not os.path.exists(self.vis_folder):
            os.mkdir(self.vis_folder)

    @property
    def modality_length(self):
        length_list = []
        for idx in range(len(self.datas)):
            length_list.append(100)
        return length_list * self.repeats

    def __len__(self):
        return len(self.datas) * self.repeats

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

    def sort_masks_by_area(self, masks):
        areas = []
        for mask in masks:
            area = np.sum(mask)
            areas.append(area)
        indexes = np.argsort(np.array(areas))[::-1]  # sort the mask from large area to small area
        return indexes

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

    def visualize(self, image_path, masks, conversations, prompt_tokens):
        text = ""
        for index in prompt_tokens:
            text += str(index) + ", "
        text += "\n"
        text += "conversation:\n"
        for item in conversations:
            text += "{}: {}\n".format(item['from'], item['value'])

        txt_path = os.path.join(self.vis_folder, f"{self.vis_idx}.txt")
        with open(txt_path, 'w') as f:
            f.write(text)

        for i, mask in enumerate(masks):
            mask_save_path = os.path.join(self.vis_folder, f"{self.vis_idx}-{i}.png")
            image = Image.open(image_path).convert('RGB')
            show_mask_pred(image, mask[None, :, :], save_dir=mask_save_path)

        self.vis_idx += 1
        return

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

        _vision_vp_tokens = np.zeros((w*h), dtype=np.int64) - 2  # -2 for none vp index
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
        index = index % len(self.datas)
        data_dict = self.datas[index]
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


def show_mask_pred(image, masks, save_dir='./output.png'):
    from PIL import Image
    import numpy as np

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
              (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (128, 128, 255)]

    _mask_image = np.zeros((masks.shape[1], masks.shape[2], 3), dtype=np.uint8)
    print(masks.shape)
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        _mask_image[:, :, 0] = _mask_image[:, :, 0] + mask.astype(np.uint8) * color[0]
        _mask_image[:, :, 1] = _mask_image[:, :, 1] + mask.astype(np.uint8) * color[1]
        _mask_image[:, :, 2] = _mask_image[:, :, 2] + mask.astype(np.uint8) * color[2]


    image = np.array(image)
    image = image * 0.5 + _mask_image * 0.5
    image = image.astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_dir)

    return

class PixelCapVisualPromptDataset_IDE_OR(PixelCapVisualPromptDataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"

    VP_FEAT_TOKEN = "<vp_feat>"
    VP_SIGN = "<VP_SIGN>"

    def __init__(self,
                 image_folder,
                 pano_rgb_folder,
                 pano_json,
                 special_tokens=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 patch_size=32,
                 add_cls=False,
                 prompt_numbers=10,
                 repeats=1,
                 **kwargs):
        super(PixelCapVisualPromptDataset_IDE_OR, self).__init__(
            image_folder=image_folder,
            pano_rgb_folder=pano_rgb_folder,
            pano_json=pano_json,
            special_tokens=special_tokens,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            max_length=max_length,
            num_classes_per_sample=num_classes_per_sample,
            patch_size=patch_size,
            add_cls=add_cls,
            prompt_numbers=prompt_numbers,
            repeats=repeats,
            **kwargs
        )

    def sort_masks_by_area(self, masks):
        areas = []
        for mask in masks:
            area = np.sum(mask)
            areas.append(area)
        indexes = np.argsort(np.array(areas))[::-1]  # sort the mask from large area to small area
        return indexes

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

        assert conversation[0]['from'] == 'human'
        vp_prefix = "There are some marked objects: "
        for i in range(len(masks)):
            vp_prefix += f"{selected_prompt_tokens[i]}: {self.VP_SIGN}_{i}, "
        vp_prefix = vp_prefix[:-2] + ".\n"
        conversation[0]['value'] = vp_prefix + conversation[0]['value']

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

    def visualize(self, image_path, masks, conversations, prompt_tokens):
        text = ""
        for index in prompt_tokens:
            text += str(index) + ", "
        text += "\n"
        text += "conversation:\n"
        for item in conversations:
            text += "{}: {}\n".format(item['from'], item['value'])

        txt_path = os.path.join(self.vis_folder, f"{self.vis_idx}.txt")
        with open(txt_path, 'w') as f:
            f.write(text)

        for i, mask in enumerate(masks):
            mask_save_path = os.path.join(self.vis_folder, f"{self.vis_idx}-{i}.png")
            image = Image.open(image_path).convert('RGB')
            show_mask_pred(image, mask[None, :, :], save_dir=mask_save_path)

        self.vis_idx += 1
        return

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

        _vision_vp_tokens = np.zeros((w*h), dtype=np.int64) - 2  # -2 for none vp index
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
        index = index % len(self.datas)
        data_dict = self.datas[index]
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

            data_dict['conversations'], vp_feat_token_indices = self.insert_vp_feat(data_dict['conversations'], masks_prompt_patches)

            token_dict = self.get_inputid_labels_vp_feat(
                data_dict['conversations'], image_token_str, image_token_patch_indices, vp_token_indices, vp_feat_token_indices)
            # out_data_dict.update(token_dict)
            out_data_dict.update(token_dict)
        else:
            out_data_dict['patch_nums_per_images'] = (0, 0)
            token_dict = self.get_inputid_labels_vp_feat(
                data_dict['conversations'], "", [], [], [])
            out_data_dict.update(token_dict)
        return out_data_dict

    def insert_vp_feat(self, conversations, mask_patches):
        vp_feat_indices = []
        for i, mask_patch in enumerate(mask_patches):
            h, w = mask_patch.shape
            _replace_str = self.VP_FEAT_TOKEN * torch.sum(mask_patch).item()
            _indices = np.arange(h * w)[mask_patch.flatten().numpy().astype(np.bool_)]
            _indices = _indices.tolist()
            vp_feat_indices += _indices
            conversations[0]["value"] = conversations[0]["value"].replace(f"{self.VP_SIGN}_{i}", _replace_str)
        return conversations, vp_feat_indices

    def get_inputid_labels_vp_feat(self, conversations, image_token_str, image_token_patch_indices, vp_token_indices, vp_feat_token_indices) -> dict:
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

        length = len(input_ids)
        vp_feat_indices = np.array([-1] * length)
        vp_feat_indices[np.array(input_ids) == self.tokenizer(self.VP_FEAT_TOKEN, add_special_tokens=False).input_ids[0]] = np.array(vp_feat_token_indices)
        # assert len(vp_feat_indices) == len(vp_patch_indices), [len(vp_feat_indices), len(vp_patch_indices)]
        # print(self.tokenizer.decode(
        #     input_ids, skip_special_tokens=False).strip())
        return {'input_ids': input_ids, 'labels': labels,
                'vision_patch_indices': token_patch_indices,
                'vision_start_end': vision_start_end,
                'vp_token_indices': vp_patch_indices,
                'vp_feat_indices': vp_feat_indices,
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



class PixelCapAgumentedVisualPromptDataset(PixelCapVisualPromptDataset):
    def __init__(self,
                 image_folder,
                 pano_rgb_folder,
                 pano_json,
                 special_tokens=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 patch_size=32,
                 add_cls=False,
                 prompt_numbers=10,
                 repeats=1,
                 prompt_ratios={"mask": 0.4, "box": 0.3, "point": 0.3},
                 **kwargs):
        super(PixelCapAgumentedVisualPromptDataset, self).__init__(
            image_folder=image_folder,
            pano_rgb_folder=pano_rgb_folder,
            pano_json=pano_json,
            special_tokens=special_tokens,
            prompt_template=prompt_template,
            tokenizer=tokenizer,
            max_length=max_length,
            num_classes_per_sample=num_classes_per_sample,
            patch_size=patch_size,
            add_cls=add_cls,
            prompt_numbers=prompt_numbers,
            repeats=repeats,
            **kwargs
        )
        self.prompt_ratios = prompt_ratios
        overall_ratio = 0.0
        for prompt_type in prompt_ratios.keys():
            assert prompt_type in ["mask", "box", "point"]
            assert prompt_ratios[prompt_type] > 0
            overall_ratio += prompt_ratios[prompt_type]
        assert overall_ratio != 0
        for prompt_type in self.prompt_ratios.keys():
            self.prompt_ratios[prompt_type] /= overall_ratio

    def select_prompt_type(self):
        seed = random.random()
        threshold = 0.0
        prompt_type = "mask"
        for prompt_type in self.prompt_ratios.keys():
            threshold += self.prompt_ratios[prompt_type]
            if seed <= threshold:
                return prompt_type
        return prompt_type

    def _parse_annotations(self, ann_info):
        ann_info = super()._parse_annotations(ann_info)
        if ann_info is None:
            return None
        masks = ann_info["masks"]
        agumented_mask = []
        for mask in masks:
            if np.sum(mask) != 0 and not has_multiple_connected_components(mask):
                # do agu for visual prompt
                prompt_type = self.select_prompt_type()
                if prompt_type == "mask":
                    agumented_mask.append(mask)
                elif prompt_type == "box":
                    box_mask = maskPrompt2boxPrompt(mask)
                    agumented_mask.append(box_mask)
                elif prompt_type == "point":
                    point_mask = maskPrompt2pointPrompt(mask, min_distance=self.patch_size)
                    agumented_mask.append(point_mask)
                else:
                    raise NotImplementedError
            else:
                agumented_mask.append(mask)
        ann_info["masks"] = agumented_mask
        return ann_info



