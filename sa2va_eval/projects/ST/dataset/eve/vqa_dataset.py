import copy
import json
import os
from mmengine import print_log
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from projects.ST.eve.mm_utils import process_images
from projects.ST.eve.constants import IMAGE_TOKEN_INDEX

class LLaVADataset(Dataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 image_preprocessor,
                 prompt_template,
                 special_tokens=None,
                 image_folder=None,
                 max_length=8192,
                 patch_size=32,
                 ):
        self.tokenizer = BUILDER.build(tokenizer)
        self.image_preprocessor = BUILDER.build(image_preprocessor)
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self._system = ''

        self.patch_size = patch_size

        self.image_folder = image_folder
        self.template = prompt_template
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
        out_data_dict = {}
        if data_dict.get('image', None) is not None:
            image_file = os.path.join(self.image_folder, data_dict['image'])
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
        # print(out_data_dict["input_ids"], out_data_dict['visual_prompt_indexes'])
        return out_data_dict

    def _rand_another(self) -> int:
        return np.random.randint(0, len(self.data))

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
        return {'input_ids': input_ids, 'labels': labels,}

