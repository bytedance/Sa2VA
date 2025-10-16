# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from .configuration_internlm2 import InternLM2Config
import transformers
from transformers import LlamaConfig, Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig
from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class Sa2VAChatConfig(PretrainedConfig):
    model_type = 'sa2va_chat'

    def __init__(
            self,
            vision_config=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            ps_version='v1',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            **kwargs):
        super().__init__(**kwargs)
        
        print("self.architectures:", self.architectures)
        print("vision_config:", vision_config)
        
        if self.architectures is None:
            self.architectures = ['Qwen2_5_VLForConditionalGeneration']

        if vision_config is None or vision_config.get('architectures') is None:
            if self.architectures[0] == 'Qwen2_5_VLForConditionalGeneration':
                vision_config = {'architectures': ['Qwen2_5_VisionTransformerPretrainedModel']}
            else:
                vision_config = {'architectures': ['InternVisionModel']}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if llm_config is None:
            if self.architectures[0] == 'Qwen2_5_VLForConditionalGeneration':
                llm_config = {'architectures': ['Qwen2ForCausalLM']}
            else:
                llm_config = {'architectures': ['InternLM2ForCausalLM']}

            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

        if vision_config.get('architectures')[0] == 'Qwen2_5_VisionTransformerPretrainedModel':
            self.vision_config = Qwen2_5_VLVisionConfig(**vision_config)
        else:
            self.vision_config = InternVisionConfig(**vision_config)

        if llm_config.get('architectures')[0] == 'LlamaForCausalLM':
            self.llm_config = LlamaConfig(**llm_config)
        elif llm_config.get('architectures')[0] == 'InternLM2ForCausalLM':
            self.llm_config = InternLM2Config(**llm_config)
        elif llm_config.get('architectures')[0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2Config(**llm_config)
        elif llm_config.get('architectures')[0] == 'Qwen2ForCausalLM':
            self.llm_config = Qwen2_5_VLConfig(**llm_config)
        elif llm_config.get('architectures')[0] == 'Qwen3ForCausalLM':
            assert transformers.__version__ >= '4.56.0', 'Please upgrade transformers to >=4.56.0 for Qwen3 support.'
            from transformers import Qwen3Config
            self.llm_config = Qwen3Config(**llm_config)
        else:
            raise ValueError('Unsupported architecture: {}'.format(llm_config.get('architectures')[0]))
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

        self.hidden_size = self.llm_config.hidden_size
        self.tie_word_embeddings = False

        logger.info(f'vision_select_layer: {self.select_layer}')
        logger.info(f'ps_version: {self.ps_version}')
        logger.info(f'min_dynamic_patch: {self.min_dynamic_patch}')
        logger.info(f'max_dynamic_patch: {self.max_dynamic_patch}')

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['pad2square'] = self.pad2square
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch

        return output
