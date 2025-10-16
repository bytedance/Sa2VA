from collections import OrderedDict
from importlib.resources import path
from typing import Dict, Optional, Union, List


import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLProcessor
from peft import get_peft_model, prepare_model_for_kbit_training


from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine import print_log

class Qwen2_5_VL(BaseModel):
    r"""
    Qwen2.5-VL: Adapter for the Qwen2.5-VL model.
    Goal: Enable the training within the xtuner framework.
    """
    def __init__(
            self,
            model_path: str,
            freeze_llm: bool = False,
            freeze_visual_encoder: bool = False,
            llm_lora: Optional[dict] = None,
            pretrained_pth: Optional[str] = None
        ):
        super().__init__()

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None


        # Note:
        # force to use flash_attention_2 and bfloat16 for training Qwen2.5-VL
        # for better acceleration and memory saving.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )

        # self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # "<|image_pad|>" is used to pad the image tokens to a fixed length.
        # This is consistent in the qwen2.5-vl model.
        img_context_token_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        self.img_context_token_id = img_context_token_id

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.visual.requires_grad_(False)


        if self.use_llm_lora:
            self.llm_lora_config = llm_lora
            print_log(f'Qwen2_5_VL: Using Lora for the LLM with config {self.llm_lora_config} (delay the lora please call manual)', logger='current')

        self.tokenizer = None
        self.processor = None

    def add_special_tokens(self, tokenizer, special_tokens: List[str]) -> None:
        """Add special tokens to the tokenizer and resize embeddings if needed."""
        print_log(f'{self.__class__.__name__}:add_special_tokens [Before] The total number of tokens is now {len(tokenizer)}', logger='current')
        num_new_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.model.resize_token_embeddings(len(tokenizer))
            print_log(f'{self.__class__.__name__}:add_special_tokens Added {num_new_tokens} special tokens', logger='current')
            print_log(f'{self.__class__.__name__}:add_special_tokens [After] The total number of tokens is now {len(tokenizer)}', logger='current')
        self.tokenizer = tokenizer

    def _init_processor(self, image_processor, video_processor):
        self.processor = Qwen2_5_VLProcessor(
            image_processor=image_processor,
            tokenizer=self.tokenizer,
            video_processor=video_processor
        )
        
    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model = prepare_model_for_kbit_training(self.model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            _target_modules = []
            for name, module in self.model.language_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    _target_modules.append('language_model.' + name)
            lora_config.target_modules = _target_modules
        self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()


    def manual_prepare_llm_for_lora(self):
        if self.use_llm_lora:
          self._prepare_llm_for_lora(self.llm_lora_config)


    def get_embedding_size(self):
        return self.model.config.text_config.hidden_size


    def forward(self,
                data: Dict[str, torch.Tensor],
                data_samples: Optional[list] = None,
                mode: str = 'loss') -> Union[Dict[str, torch.Tensor], list]:
        assert mode == 'loss', f'Only support loss mode in {self.__class__.__name__}, but got {mode}'
        pixel_values: List[torch.Tensor] = data['pixel_values']
        pixel_values = torch.cat(pixel_values, dim=0)
        image_grid_thw = data['image_grid_thw']
        image_grid_thw = torch.cat(image_grid_thw, dim=0)
        
        
        # DO NOT ENTER POSTION EMBEDDING HERE; Qwen2.5-VL will handle it inside (M-ROPE)
        output = self.model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            labels=data['labels'],
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            # output hidden states for potential further usage
            output_hidden_states=True,
        )
        return output


    def state_dict(self, *args, **kwargs):
        # filter out the untrainable parameters
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        to_return.update(get_peft_model_state_dict(self.model, state_dict=state_dict))
        return to_return

    def init_weights(self):
        # Always load from pretrained weights
        pass

if __name__ == "__main__":
    from peft import LoraConfig
    import copy
    model = Qwen2_5_VL(
        model_path="pretrained/qwen2_5vl/Qwen2.5-VL-7B-Instruct/",
        freeze_llm=False,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=256,
            lora_alpha=512,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            modules_to_save=['lm_head', 'embed_tokens'],
            target_modules=None,
        ),
    )


    tokenizer = AutoTokenizer.from_pretrained("pretrained/qwen2_5vl/Qwen2.5-VL-7B-Instruct/", trust_remote_code=True)
    model.add_special_tokens(tokenizer, special_tokens=['[SEG]'])

    model.manual_prepare_llm_for_lora()
    model = model.to('cuda')

    processor = AutoProcessor.from_pretrained("pretrained/qwen2_5vl/Qwen2.5-VL-7B-Instruct/", trust_remote_code=True)
    from qwen_vl_utils import process_vision_info
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    video_inputs = [copy.deepcopy(image_inputs[0]) for _ in range(10)]  # mock video with 10 frames
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")


    mock_data_dict = {
        'input_ids': inputs['input_ids'], # (1, 3602) torch.int64
        'attention_mask': inputs['attention_mask'], #  (1, 3602) torch.int64, all 1
        'labels': inputs['input_ids'], # (1, 3602) torch.int64
        'pixel_values': [inputs['pixel_values']], # torch.Size([14308, 1176]) torch.float32 torch.Size([11008, 1536]) for qwen3
        'image_grid_thw': [inputs['image_grid_thw']] # value: 1, 98, 146 (not shape) torch.int64; 1, 86. 128 for qwen3
    }


    output = model(mock_data_dict, mode='loss')
    print(output)