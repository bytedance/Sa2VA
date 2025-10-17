import copy
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..eve_arch import EVEMetaModel, EVEMetaForCausalLM
from .qwen2 import Qwen2Model, Qwen2Config, Qwen2ForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast


class EVEQwen2Config(Qwen2Config):
    model_type = "eve-qwen2"


class EVEQwen2Model(EVEMetaModel, Qwen2Model):
    config_class = EVEQwen2Config

    def __init__(self, config: Qwen2Config):
        super(EVEQwen2Model, self).__init__(config)


class EVEQwen2ForCausalLM(Qwen2ForCausalLM, EVEMetaForCausalLM):
    config_class = EVEQwen2Config

    def __init__(self, config):
        super(Qwen2ForCausalLM, self).__init__(config)
        self.config = config
        self.model = EVEQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.vp_embed = nn.Linear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
        )

        # Initialize weights and apply final processing
        # self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            visual_prompt_indexes = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                visual_token_mask,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                visual_prompt_indexes=visual_prompt_indexes,
            )


        ret =  super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            visual_token_mask=visual_token_mask,

        )
        ret.st_input_ids = copy.deepcopy(labels)
        return ret

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, attention_mask=None,
                                      **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        return _inputs
