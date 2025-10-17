import copy
import json
import random
import os
from mmengine import print_log
from PIL import Image
import numpy as np
from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from .utils import convert_image_to_patches, convert_mask_to_patches
from torch.utils.data import Dataset
from pycocotools import mask as mask_utils

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

class SamVisualPromptDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"

    def __init__(self,
                 image_folder,
                 json_file,
                 special_tokens=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 patch_size=32,
                 add_cls=False,
                 prompt_numbers=10,
                 **kwargs):
        self._system = ''
        self.prompt_numbers = prompt_numbers
        self.num_classes_per_sample = num_classes_per_sample

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
        self.json_file = json_file

        self.datas = self.read_json()

        self.template = prompt_template
        self.template['INSTRUCTION'] = PROMPT_TMPL
        self.template['SUFFIX'] = '<|endoftext|>'
        self.max_length = max_length
        self._max_refetch = 1000

        print("SAM visual prompt dataset, include {} items.".format(len(self.datas)))

        self._long_limit = " Please provide as detailed as possible."

        self.vis_idx = 0
        self.vis_folder = "./data_vis/"
        if not os.path.exists(self.vis_folder):
            os.mkdir(self.vis_folder)

    @property
    def modality_length(self):
        length_list = []
        for idx in range(len(self.datas)):
            length_list.append(100)
        return length_list

    def __len__(self):
        return len(self.datas)

    def read_json(self):
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        ret = []
        # process and split the annotation
        for image_annotation in data:
            image_name = image_annotation['image_name']
            objects_annotations = image_annotation["object_annotations"]

            random.shuffle(objects_annotations)
            n_split = len(objects_annotations) // self.num_classes_per_sample
            if len(objects_annotations) > n_split * self.num_classes_per_sample:
                n_split += 1

            for i_split in range(n_split):
                start_idx = i_split * self.num_classes_per_sample
                end_idx = (i_split + 1) * self.num_classes_per_sample
                end_idx = min(end_idx, len(objects_annotations))
                _item = {'image_name': image_name, 'object_annotations': objects_annotations[start_idx: end_idx]}
                ret.append(_item)
        random.shuffle(ret)
        return ret

    def sort_masks_by_area(self, masks):
        areas = []
        for mask in masks:
            area = np.sum(mask)
            areas.append(area)
        indexes = np.argsort(np.array(areas))[::-1]  # sort the mask from large area to small area
        return indexes

    def _parse_annotations(self, ann_info):
        image_name = ann_info['image_name']
        image_path = os.path.join(self.image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        width, height = image.size
        objects_annotations = ann_info["object_annotations"]

        masks = [object_annotation["segmentation"] for object_annotation in objects_annotations]
        masks = [mask_utils.decode(mask) for mask in masks]
        descriptions = [object_annotation["caption"] for object_annotation in objects_annotations]

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

        for i, (mask, obj_description) in enumerate(zip(masks, descriptions)):
            obj_description = obj_description.replace("highlighted", "").replace("highlight", "")\
                .replace("by a yellow edge", "").replace("by a yellow contour", "").replace("yellow contour", "")\
                .replace("yellow edge", "").replace("yellow outline", "")
            if none_prompt and random.random() < 0.05:
                question = random.choice(DETAILED_QUESTIONS).replace("<region>", not_selected_prompt_tokens[0])
                conversation.append({'from': 'human', 'value': question + self._long_limit})
                conversation.append({'from': 'gpt', 'value': f"{not_selected_prompt_tokens[0]} is not in the image."})
                none_prompt = False
            question = random.choice(DETAILED_QUESTIONS).replace("<region>", selected_prompt_tokens[i])
            conversation.append({'from': 'human', 'value': question + self._long_limit})
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

