import torch
from transformers import AutoTokenizer, AutoConfig, AutoModel, CLIPImageProcessor
from PIL import Image
from .base import BaseModel
from ..smp import *
from ..dataset import DATASET_TYPE
import pandas as pd
import string
import transformers
import re

from mmengine.config import Config, DictAction
from xtuner.registry import BUILDER
from mmengine.fileio import PetrelBackend, get_file_backend
from xtuner.model.utils import guess_load_checkpoint

class STChat(BaseModel):
    def __init__(self,
                 config_file,
                 pth_model=None,
                 **kwargs):

        # load config
        cfg = Config.fromfile(config_file)
        cfg.model.pretrained_pth = None

        model = BUILDER.build(cfg.model)

        if pth_model is not None:
            backend = get_file_backend(pth_model)
            if isinstance(backend, PetrelBackend):
                from xtuner.utils.fileio import patch_fileio
                with patch_fileio():
                    state_dict = guess_load_checkpoint(pth_model)
            else:
                state_dict = guess_load_checkpoint(pth_model)

            model.load_state_dict(state_dict, strict=False)
        model.cuda()
        model.eval()
        model.preparing_for_generation(metainfo={})
        self.model = model

        print(f'Load PTH model from {pth_model}')
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
                # prompt = question + ' Answer the question using a single word or phrase.'
                prompt = question # to align weixian setting
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



    def generate_v2(self, message, dataset=None):
        image_num = len([x for x in message if x['type'] == 'image'])
        if image_num == 1:
            prompt = '<image>\n' + '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        elif image_num == 0:
            prompt = '\n'.join([x['value'] for x in message if x['type'] == 'text'])
        else:
            raise NotImplementedError
        if image_num == 1:
            image_path = [x['value'] for x in message if x['type'] == 'image'][0]
            ori_image = Image.open(image_path).convert('RGB')
            input_dict = {
                'image': ori_image,
                'text': prompt,
                'past_text': '',
            }
        else:
            ori_image = None
            input_dict = {
                'image': ori_image,
                'text': prompt,
                'past_text': '',
            }

        with torch.no_grad():
            response = self.model.predict_forward(
                **input_dict
            )["prediction"]\
            .replace("<|end|>", "").replace("<|endoftext|>", "")\
            .replace("<|im_end|>", "").strip()

        print(response)
        return response

    def generate_inner(self, message, dataset=None):
        return self.generate_v2(message, dataset)