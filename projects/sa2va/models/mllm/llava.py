from collections import OrderedDict
from typing import Dict, Optional, Union, List


import torch
from transformers import LlavaForConditionalGeneration, AutoTokenizer
from peft import PeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training


from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine import print_log

class LlavaVLM(BaseModel):
    r"""
    LLaVA-1.5: Adapter for the HF LLaVA model (CLIP-ViT-L-336 + Vicuna).
    Goal: Enable the training within the xtuner framework, so Sa2VA can be
    compared against LISA-style baselines that use the same backbone.
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
        # force to use flash_attention_2 and bfloat16 for training LLaVA
        # for better acceleration and memory saving.
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )

        self.model.gradient_checkpointing_enable()

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # "<image>" (id 32000) already exists in the llava-hf tokenizer; the model
        # scatters vision features into these placeholder positions.
        self.img_context_token_id = self.model.config.image_token_id

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.vision_tower.requires_grad_(False)
        # The multi-modal projector is NOT covered by the PEFT state_dict filter
        # below, so any training on it would be silently dropped at save time.
        # Keep it frozen (its weights come from the pretrained checkpoint).
        self.model.multi_modal_projector.requires_grad_(False)


        if self.use_llm_lora:
            self.llm_lora_config = llm_lora
            print_log(f'LlavaVLM: Using Lora for the LLM with config {self.llm_lora_config} (delay the lora please call manual)', logger='current')

        self.tokenizer = None

    def add_special_tokens(self, tokenizer, special_tokens: List[str]) -> None:
        """Add special tokens to the tokenizer and resize embeddings if needed.

        Note: the llava-hf embedding matrix is padded to 32064 while the
        tokenizer has 32002 entries; adding 5 Sa2VA tokens resizes the matrix
        down to 32007 (truncates unused pad rows) — functionally fine, and
        convert_to_hf.py sets vocab_size = len(tokenizer) accordingly.
        """
        print_log(f'{self.__class__.__name__}:add_special_tokens [Before] The total number of tokens is now {len(tokenizer)}', logger='current')
        num_new_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.model.resize_token_embeddings(len(tokenizer))
            print_log(f'{self.__class__.__name__}:add_special_tokens Added {num_new_tokens} special tokens', logger='current')
            print_log(f'{self.__class__.__name__}:add_special_tokens [After] The total number of tokens is now {len(tokenizer)}', logger='current')
        self.tokenizer = tokenizer

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
        # per-sample (n_i, 3, 336, 336) -> (sum n_i, 3, 336, 336)
        pixel_values = torch.cat(pixel_values, dim=0).to(self.model.dtype)

        # Drop all-zero placeholder frames inserted by datasets for text-only
        # samples; they have NO matching <image> tokens in input_ids, and HF
        # llava hard-errors on a token/feature count mismatch. Real images
        # never normalize to all-zero under the CLIP mean/std.
        keep = pixel_values.flatten(1).abs().sum(-1) != 0
        pixel_values = pixel_values[keep]
        if pixel_values.shape[0] == 0:
            pixel_values = None

        output = self.model(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            labels=data['labels'],
            pixel_values=pixel_values,
            # output hidden states for the [SEG] embedding extraction
            output_hidden_states=True,
        )
        return output


    def state_dict(self, *args, **kwargs):
        # filter out the untrainable parameters
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        if isinstance(self.model, PeftModelForCausalLM):
            to_return.update(get_peft_model_state_dict(self.model, state_dict=state_dict))
        else:
            to_return.update(state_dict)
        return to_return

    def init_weights(self):
        # Always load from pretrained weights
        pass

if __name__ == "__main__":
    from peft import LoraConfig
    model = LlavaVLM(
        model_path="pretrained/llava/llava-1.5-7b-hf/",
        freeze_llm=True,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            modules_to_save=['lm_head', 'embed_tokens'],
            target_modules=None,
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained("pretrained/llava/llava-1.5-7b-hf/", trust_remote_code=True)
    model.add_special_tokens(tokenizer, special_tokens=['[SEG]', '<p>', '</p>', '<vp>', '</vp>'])

    model.manual_prepare_llm_for_lora()
    model = model.to('cuda')

    # mock one image sample: 576 <image> tokens + a short prompt
    image_token_id = model.img_context_token_id
    prompt = "USER: " + "<image>" * 576 + "\nDescribe this image. ASSISTANT: A cat."
    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to('cuda')
    pixel_values = torch.randn(1, 3, 336, 336, dtype=torch.bfloat16, device='cuda')

    mock_data_dict = {
        'input_ids': input_ids,
        'attention_mask': torch.ones_like(input_ids),
        'labels': input_ids.clone(),
        'pixel_values': [pixel_values],
    }

    # training runs under AmpOptimWrapper(dtype='bfloat16') autocast; mirror it
    # here (prepare_model_for_kbit_training upcasts params to fp32, and FA2
    # only supports fp16/bf16 inputs)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = model(mock_data_dict, mode='loss')
    print(output.loss, output.hidden_states[-1].shape)
