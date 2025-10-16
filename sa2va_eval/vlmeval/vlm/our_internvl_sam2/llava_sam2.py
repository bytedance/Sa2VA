import torch
from xtuner.model.utils import get_peft_model_state_dict

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from transformers import GenerationConfig
from projects.sa2va.models.preprocess.image_resize import DirectResize

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from types import MethodType

from xtuner.model.utils import guess_load_checkpoint
from mmengine.model import BaseModel

from transformers import AutoTokenizer
from .internvl import InternVL_Slowfast

class VideoLLaVASAMModel(BaseModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 special_tokens=['[SEG]', ],
                 # for slow fast arch
                 fast_pool=False,
                 fast_pool_size=2,
                 metainfo={},
                 phi3=True,
                 fast_frames=50,
                 slow_frames=5,
                 ):
        super().__init__()
        self.special_tokens = special_tokens
        if 'special_tokens' not in mllm.keys():
            mllm.special_tokens = special_tokens
        self.mllm = InternVL_Slowfast(**mllm)

        self.fast_pool = fast_pool
        self.fast_pool_size = fast_pool_size
        if hasattr(self.mllm, '_post_init'):
            self.mllm._post_init(
                fast_pool_size=self.fast_pool_size,
                fast_pool=self.fast_pool
            )
        else:
            print("No _post_init() in mllm !!!")

        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer)
        self._add_special_tokens()

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        if fast_pool:
            self.fast_token_idx = self.tokenizer("<FAST_IMG_CONTEXT>", add_special_tokens=False).input_ids[0]
        else:
            self.fast_token_idx = None

        self.preparing_for_generation(metainfo)
        self.phi3 = phi3

        self.fast_frames = fast_frames
        self.slow_frames = slow_frames


    def _add_special_tokens(self):
        special_tokens = self.special_tokens
        num_new_tokens = self.tokenizer.add_tokens(
            special_tokens, special_tokens=True)

        assert isinstance(self.mllm, InternVL_Slowfast)

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        from collections import OrderedDict

        to_return = OrderedDict()
        # Step 1. visual_encoder
        if self.mllm.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.mllm.model.vision_model, state_dict=state_dict))
        elif not self.mllm.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.mllm.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.mllm.model.language_model, state_dict=state_dict))
        elif not self.mllm.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mlp1.' in k})
        # Step 4. mask decoder of grounding model (SAM/SAM2)
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'mask_decoder' in k})

        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'text_hidden_fcs.' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'lm_head.weight' in k})
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'embed_tokens.weight' in k})
        return to_return

    def preparing_for_generation(self, metainfo, **kwargs):
        # set stop criteria and generation configs for model
        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
        self.bot_name = 'BOT'
        if 'template' in metainfo.keys():
            template = metainfo['template']
        else:
            template = PROMPT_TEMPLATE['phi3_chat']
        self.template = template
        stop_words = []
        stop_words += template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=(
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        )
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True

        self.mllm.to(self.torch_dtype)
        self.text_hidden_fcs.to(self.torch_dtype)

        # for sam image processor
        self.extra_image_processor = DirectResize(target_length=1024, )
        # for multi image process
        self.min_dynamic_patch = 1
        self.max_dynamic_patch = 12
        self.downsample_ratio = 0.5
        self.image_size = 448
        self.use_thumbnail = True
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        self.IMG_START_TOKEN = '<img>'
        self.IMG_END_TOKEN = '</img>'

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        # change phi3 prepare for generation fuction
        if self.phi3:
            self.mllm.model.language_model.prepare_inputs_for_generation = MethodType(prepare_inputs_for_generation, self.mllm.model.language_model)
        return

    def predict_forward(
            self,
            text_prompt,
            ori_image=None,
            ori_image_fast=None,
            **kwargs
    ):

        input_dict = {}
        # prepare images
        assert ori_image is not None, "InternVL2 only support process the image from scratch !!!"
        assert isinstance(ori_image, list)

        all_pixel_values = ori_image.to(self.torch_dtype)
        input_dict['pixel_values'] = all_pixel_values

        if ori_image_fast is None:
            fast_token_idx = None
        else:
            fast_token_idx = self.fast_token_idx
        fast_pixel_values = ori_image_fast
        if fast_pixel_values is not None:
            fast_pixel_values = fast_pixel_values.to(dtype=self.torch_dtype)

        text_prompt = text_prompt
        input_text = ''
        input_text += self.template['INSTRUCTION'].format(
            input=text_prompt, round=1, bot_name=self.bot_name)

        ids = self.tokenizer.encode(input_text)
        ids = torch.tensor(ids).cuda().unsqueeze(0)

        attention_mask = torch.ones_like(ids, dtype=torch.bool)

        mm_inputs = {
            'pixel_values': input_dict['pixel_values'],
            'input_ids': ids,
            'attention_mask': attention_mask,
            'position_ids': None,
            'past_key_values': None,
            'labels': None,
            'fast_pixel_values': fast_pixel_values,
            'fast_token_idx': fast_token_idx,
        }

        generate_output = self.mllm.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0], skip_special_tokens=True).strip()

        return predict

def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
):
    if past_key_values is not None:
        if isinstance(past_key_values, Cache):
            cache_length = past_key_values.get_seq_length()
            past_length = past_key_values.seen_tokens
            max_cache_length = past_key_values.get_max_length()
        else:
            cache_length = past_length = past_key_values[0][0].shape[2]
            max_cache_length = None

        # Keep only the unprocessed tokens:
        # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
        # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
        # input)
        if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
            input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
        # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
        # input_ids based on the past_length.
        elif past_length < input_ids.shape[1]:
            input_ids = input_ids[:, past_length:]
        # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

        # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
        if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
        ):
            attention_mask = attention_mask[:, -max_cache_length:]

    position_ids = kwargs.get('position_ids', None)
    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and (past_key_values is None or len(past_key_values)==0):
        model_inputs = {'inputs_embeds': inputs_embeds}
    else:
        model_inputs = {'input_ids': input_ids}

    model_inputs.update(
        {
            'position_ids': position_ids,
            'past_key_values': past_key_values,
            'use_cache': kwargs.get('use_cache'),
            'attention_mask': attention_mask,
        }
    )
    return model_inputs


