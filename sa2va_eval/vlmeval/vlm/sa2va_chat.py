import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor, AutoModelForCausalLM, AutoProcessor
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import pandas as pd
import string
import transformers
import re

def split_model(model_name):
    import math
    device_map = {}
    num_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = num_gpus // world_size

    num_layers = {'InternVL2-8B': 32, 'InternVL2-26B': 48,
                  'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80,
                  'InternVL2_5-8B': 32, 'InternVL2_5-26B': 48,
                  'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as 0.8 GPU.
    num_layers_per_gpu = math.ceil(num_layers / (num_gpus - 0.2))
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.8)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = rank + world_size * i
            layer_cnt += 1
    device_map['vision_model'] = rank
    device_map['mlp1'] = rank
    device_map['language_model.model.tok_embeddings'] = rank
    device_map['language_model.model.embed_tokens'] = rank
    device_map['language_model.output'] = rank
    device_map['language_model.model.norm'] = rank
    device_map['language_model.lm_head'] = rank
    device_map[f'language_model.model.layers.{num_layers - 1}'] = rank
    device_map['grounding_encoder'] = rank
    device_map['text_hidden_fcs'] = rank
    return device_map

class Sa2VAChat(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path='OMG-Research/Sa2VA-4B', load_in_8bit=False,
                 model_split=False, model_split_name=None,
                 **kwargs):
        assert model_path is not None
        assert version_cmp(transformers.__version__, '4.56.1', 'ge')

        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        if 'qwen' in model_path.lower() and 'sa2va' in model_path.lower():
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        else:
            self.processor = None
        # Regular expression to match the pattern 'Image' followed by a number, e.g. Image1
        self.pattern = r'Image(\d+)'
        # Replacement pattern to insert a hyphen between 'Image' and the number, e.g. Image-1
        self.replacement = r'Image-\1'

        # Convert InternVL2 response to dataset format
        # e.g. Image1 -> Image-1

        # Regular expression to match the pattern 'Image-' followed by a number
        self.reverse_pattern = r'Image-(\d+)'
        # Replacement pattern to remove the hyphen (Image-1 -> Image1)
        self.reverse_replacement = r'Image\1'

        if model_split:
            assert model_split_name is not None
            device_map = split_model(model_split_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_8bit=load_in_8bit,
                device_map=device_map,
            ).eval()

        else:
            device = torch.cuda.current_device()
            self.device = device
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                load_in_8bit=load_in_8bit).eval()
            if not load_in_8bit:
                self.model = self.model.to(device)

        #self.image_size = self.model.config.vision_config.image_size
        self.model.to(torch.bfloat16)

        self.version = 'V2.0'

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if listinstr(['MMDU', 'MME-RealWorld', 'MME-RealWorld-CN'], dataset):
            # For Multi-Turn we don't have custom prompt
            return False
        if listinstr(['MMBench-Video', 'Video-MME', 'MVBench', 'Video'], dataset):
            # For Video benchmarks we don't have custom prompt at here
            return False
        else:
            return True

    def build_multi_choice_prompt(self, line, dataset=None):
        question = line['question']
        hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
        if hint is not None:
            question = hint + '\n' + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f'\n{key}. {item}'
        prompt = question

        if len(options):
            prompt += '\n请直接回答选项字母。' if cn_string(
                prompt) else "\nAnswer with the option's letter from the given choices directly."
        else:
            prompt += '\n请直接回答问题。' if cn_string(prompt) else '\nAnswer the question directly.'

        return prompt

    def build_video_prompt(self, prompt, dataset=None, max_frames=64):
        for start in range(0, max_frames, 8):
            images_to_remove = ''.join([f'<Image-{i}>' for i in range(start + 1, start + 9)])
            prompt = prompt.replace(images_to_remove, '')
        for i in range(max_frames):
            prompt = prompt.replace(f'Image-{i + 1}', f'Frame-{i + 1}')
        if listinstr(['MMBench-Video'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
        elif listinstr(['Video-MME'], dataset):
            prompt = prompt.replace('\nAnswer:', '')
            prompt += "\nAnswer with the option's letter from the given choices directly."
        elif listinstr(['MVBench'], dataset):
            prompt = prompt.replace('Best option:(', '')

        return prompt

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        if self.version == 'V1.1':
            kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=5)
        else:
            kwargs_default = dict(do_sample=False, max_new_tokens=1024, top_p=None, num_beams=1)
        self.kwargs = kwargs_default

        if dataset is not None and DATASET_TYPE(dataset) == 'Y/N':
            question = line['question']
            if listinstr(['MME'], dataset):
                prompt = question + ' Answer the question using a single word or phrase.'
            elif listinstr(['HallusionBench'], dataset):
                prompt = question + ' Please answer yes or no. Answer the question using a single word or phrase.'
        elif dataset is not None and DATASET_TYPE(dataset) == 'MCQ':
            prompt = self.build_multi_choice_prompt(line, dataset)
        elif dataset is not None and DATASET_TYPE(dataset) == 'VQA':
            question = line['question']
            if listinstr(['MathVista', 'MathVision', 'VCR', 'MTVQA', 'MMVet', 'MathVerse'], dataset):
                prompt = question
            elif listinstr(['LLaVABench'], dataset):
                prompt = question + '\nAnswer this question in detail.'
            else:
                prompt = question + '\nAnswer the question using a single word or phrase.'
        else:
            prompt = line['question']
        message = [dict(type='text', value=prompt)]
        message.extend([dict(type='image', value=s) for s in tgt_path])
        return message

    def set_max_num(self, dataset):
        assert dataset is not None
        res_1_datasets = ['MMBench-Video', 'Video-MME', 'MVBench', 'Video']
        res_12_datasets = ['ChartQA_TEST', 'MMMU_DEV_VAL', 'MMMU_TEST', 'MME-RealWorld',
                           'MME-RealWorld', 'VCR_EN', 'VCR_ZH']
        res_18_datasets = ['DocVQA_VAL', 'DocVQA_TEST']
        res_24_datasets = ['InfoVQA_VAL', 'InfoVQA_TEST', 'OCRBench', 'HRBench4K', 'HRBench8K']
        if listinstr(res_1_datasets, dataset):
            self.max_num = 1
        elif listinstr(res_12_datasets, dataset):
            self.max_num = 12
        elif listinstr(res_18_datasets, dataset):
            self.max_num = 18
        elif listinstr(res_24_datasets, dataset):
            self.max_num = 24
        else:
            self.max_num = 6

    def generate_v2(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        if image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        elif image_num == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        else:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
            # raise NotImplementedError
        if listinstr(['Video', 'MVBench'], dataset):
            prompt = self.build_video_prompt(prompt, dataset)

        if image_num > 1:
            # raise NotImplementedError
            image_path = [x['value'] for x in message if x['type'] == 'image']
            ori_image = [Image.open(_image_path).convert('RGB') for _image_path in image_path]
            if self.processor is not None:
                    input_dict = {
                    'video': ori_image,
                    'text': prompt,
                    'past_text': '',
                    'mask_prompts': None,
                    'processor': self.processor,
                }
            else:
                input_dict = {
                    'video': ori_image,
                    'text': prompt,
                    'past_text': '',
                    'mask_prompts': None,
                    'tokenizer': self.tokenizer,
                }

        elif image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            ori_image = Image.open(image_path).convert('RGB')
            if self.processor is not None:
                input_dict = {
                    'image': ori_image,
                    'text': prompt,
                    'past_text': '',
                    'mask_prompts': None,
                    'processor': self.processor,
                }
            else:
                input_dict = {
                    'image': ori_image,
                    'text': prompt,
                    'past_text': '',
                    'mask_prompts': None,
                    'tokenizer': self.tokenizer,
                }
        else:
            ori_image = None
            if self.processor is not None:
                input_dict = {
                    'image': ori_image,
                    'text': prompt,
                    'past_text': '',
                    'mask_prompts': None,
                    'processor': self.processor,
                }
            else:
                input_dict = {
                    'image': ori_image,
                    'text': prompt,
                    'past_text': '',
                    'mask_prompts': None,
                    'tokenizer': self.tokenizer,
                }

        # input_dict = {
        #     'image': ori_image,
        #     'text': prompt,
        #     'past_text': '',
        #     'mask_prompts': None,
        #     'tokenizer': self.tokenizer,
        # }

        with torch.no_grad():
            response = self.model.predict_forward(
                **input_dict
            )['prediction']\
            .replace("<|end|>", "").replace("<|endoftext|>", "")\
            .replace("<|im_end|>", "").strip()

        print("Question:", prompt, "\nResponse:")
        return response

    def generate_inner(self, message, dataset=None):
        self.set_max_num(dataset)
        # print(f'InternVL model version: {self.version}')
        if self.version == 'V2.0':
            return self.generate_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version: {self.version}')

    def build_history(self, message):
        # Global Variables
        image_path = []
        image_cnt = 0

        def concat_tilist(tilist):
            nonlocal image_cnt  # Declare image_cnt as nonlocal to modify it
            prompt = ''
            for item in tilist:
                # Substitute the pattern in the text
                if item['type'] == 'text':
                    prompt += re.sub(self.pattern, self.replacement, item['value'])
                elif item['type'] == 'image':
                    image_cnt += 1
                    prompt += '<image>\n'
                    image_path.append(item['value'])
            return prompt

        # Only previous messages
        assert len(message) % 2 == 0
        history = []
        for i in range(len(message) // 2):
            m1, m2 = message[2 * i], message[2 * i + 1]
            assert m1['role'] == 'user' and m2['role'] == 'assistant'
            history.append((concat_tilist(m1['content']), concat_tilist(m2['content'])))

        return history, image_path, image_cnt

    def chat_inner_v2(self, message, dataset=None):

        image_cnt = 0
        if len(message) > 1:
            history, image_path, image_cnt = self.build_history(message[:-1])
        else:
            history, image_path, image_cnt = None, [], 1
        current_msg = message[-1]
        question = ''

        # If message is just text in the conversation
        if len(current_msg['content']) == 1 and current_msg['content'][0]['type'] == 'text':
            question = current_msg['content'][0]['value']
            question = re.sub(self.pattern, self.replacement, question)  # Fix pattern as per InternVL
        else:
            for msg in current_msg['content']:
                if msg['type'] == 'text':
                    question += re.sub(self.pattern, self.replacement, msg['value'])
                elif msg['type'] == 'image':
                    image_cnt += 1
                    question += '<image>\n'
                    image_path.append(msg['value'])

        if image_cnt > 1:
            raise NotImplementedError
        elif image_cnt == 1:
            ori_image = Image.open(image_path).convert('RGB')
        else:
            ori_image = None
        print(question)
        if self.processor is not None:
            input_dict = {
                'image': ori_image,
                'text': question,
                'past_text': '',
                'mask_prompts': None,
                'processor': self.processor,
            }
        else:
            input_dict = {
                'image': ori_image,
                'text': question,
                'past_text': '',
                'mask_prompts': None,
                'tokenizer': self.tokenizer,
            }

        response = self.model.predict_forward(
            **input_dict
        )['prediction'].replace("<|end|>", "").strip()

        # print(question, ' ', response)

        response = re.sub(self.reverse_pattern, self.reverse_replacement, response)

        return response

    def chat_inner(self, message, dataset=None):
        self.set_max_num(dataset)

        if self.version in ['V1.1', 'V1.2']:
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
        elif self.version == 'V1.5':
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
        elif self.version == 'V2.0':
            return self.chat_inner_v2(message, dataset)
        else:
            raise ValueError(f'Unsupported version for Multi-Turn: {self.version}')
