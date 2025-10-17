import copy
import json
import os
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX

PROMPT_TMPL = '<|im_start|>user\n{input}<|im_end|>\n'
from .utils import convert_image_to_patches
NON_VISION_TOKEN = -1

class InfinityMMDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"

    def __init__(self,
                 tokenizer,
                 data_path,
                 prompt_template,
                 special_tokens=None,
                 max_length=8192,
                 patch_size=32,
                 offline_save_path='./work_dirs/infinityMM.json',
                 add_cls=False,
                 ):
        self.add_cls = add_cls
        self.offline_save_path = offline_save_path
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

        self._system = ''

        self.template = prompt_template
        self.template['INSTRUCTION'] = PROMPT_TMPL
        self.template['SUFFIX'] = '<|endoftext|>'
        self.max_length = max_length

        self.patch_size = patch_size
        self.data = self._load_annotations(data_path)
        self._max_refetch = 1000

    def _load_annotations(self, data_path):
        if os.path.exists(self.offline_save_path):
            with open(self.offline_save_path, 'r') as f:
                ret = json.load(f)
            print(f"Load InfinityMM file list from {self.offline_save_path}, {len(ret)} items !!!")
            return ret
        sub_folders = []
        for sub_folder in os.listdir(data_path):
            if '.' not in sub_folder:
                # a folder
                if "LVIS_111k" in sub_folder:
                    # special case, have subsub folder
                    subsub_folders = os.listdir(os.path.join(data_path, sub_folder))
                    for subsub_folder in subsub_folders:
                        sub_folders.append(os.path.join(data_path, sub_folder, subsub_folder))
                else:
                    sub_folders.append(os.path.join(data_path, sub_folder))

        all_jsons = []
        for sub_folder in sub_folders:
            print(f"Processing {sub_folder} !!!")
            _files = os.listdir(sub_folder)
            _num = 0
            for _file in _files:
                if '.json' in _file:
                    _json_path = os.path.join(sub_folder, _file)
                    _num += 1
                    all_jsons.append(os.path.join(sub_folder, _file))
            print(f"Finished {sub_folder} has {_num} items.")

        with open(self.offline_save_path, 'w') as f:
            json.dump(all_jsons, f)

        return all_jsons

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def __len__(self):
        return len(self.data)

    @property
    def modality_length(self):
        self.group_length = []
        for data_dict in self.data:
            self.group_length.append(100)
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length

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
        data_path = self.data[index]

        with open(data_path, 'r') as f:
            data_dict = json.load(f)
        if 'image' in data_dict.keys():
            data_dict['image'] = data_path.replace('.json', '.jpg')

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
            # tensor, (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
            h_patches, w_patches = image_patches.shape[:2]
            out_data_dict['vision_patches'] = image_patches.flatten(0, 1).flatten(1)  # (n_patches, 3*patch_size*patch_size)
            out_data_dict['patch_nums_per_images'] = (h_patches, w_patches)

            image_token_str, image_token_len, image_token_patch_indices = \
                self.prepare_image_textual_seq_norowsep(
                image_patches.shape[0], image_patches.shape[1]
            )

            token_dict = self.get_inputid_labels(
                data_dict['conversations'], image_token_str, image_token_patch_indices)
            out_data_dict.update(token_dict)
        else:
            out_data_dict['patch_nums_per_images'] = (0, 0)
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], "", [])
            out_data_dict.update(token_dict)
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))

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

    def search_vision_tokens(self, input_ids):
        image_start_idx = self.tokenizer(self.IMG_START_TOKEN, add_special_tokens=False).input_ids[0]
        image_end_idx = self.tokenizer(self.IMG_END_TOKEN, add_special_tokens=False).input_ids[0]
        if image_start_idx not in input_ids:
            return None
        else:
            start_idx = input_ids.index(image_start_idx)
            end_idx = input_ids.index(image_end_idx)
            return [start_idx+1, end_idx]

class LLaVADataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"

    def __init__(self,
                 tokenizer,
                 data_path,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 max_length=8192,
                 patch_size=32,
                 add_cls=False,
                 ):
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

        self._system = ''

        self.patch_size = patch_size

        self.image_folder = image_folder
        self.template = prompt_template
        self.template['INSTRUCTION'] = PROMPT_TMPL
        self.template['SUFFIX'] = '<|endoftext|>'
        self.max_length = max_length

        self.data = self._load_annotations(data_path, image_folder)
        self._max_refetch = 1000

    def _load_annotations(self, data_path, image_folder=None):
        data = json.load(open(data_path))
        return data

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def __len__(self):
        return len(self.data)

    @property
    def modality_length(self):
        self.group_length = []
        for data_dict in self.data:
            self.group_length.append(100)
        return self.group_length

    @property
    def length(self):
        group_length = np.array(self.group_length)
        group_length = np.abs(group_length).tolist()
        return group_length
    
    def prepare_data(self, index):
        data_dict: dict = self.data[index]
        
        if data_dict is None:
            return None
        
        out_data_dict = {'vision_patch_idx': self.tokenizer.vis_patch_tok_id}

        if data_dict.get('image', None) is not None:
            image_file = os.path.join(self.image_folder, data_dict['image'])
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
        else:
            out_data_dict['patch_nums_per_images'] = (0, 0)
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], "", [])
            out_data_dict.update(token_dict)
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))

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

    def search_vision_tokens(self, input_ids):
        image_start_idx = self.tokenizer(self.IMG_START_TOKEN, add_special_tokens=False).input_ids[0]
        image_end_idx = self.tokenizer(self.IMG_END_TOKEN, add_special_tokens=False).input_ids[0]
        if image_start_idx not in input_ids:
            return None
        else:
            start_idx = input_ids.index(image_start_idx)
            end_idx = input_ids.index(image_end_idx)
            return [start_idx + 1, end_idx]

