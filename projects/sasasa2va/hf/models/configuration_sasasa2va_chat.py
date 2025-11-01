# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import copy

from .configuration_internlm2 import InternLM2Config
from .configuration_phi3 import Phi3Config
from transformers import AutoConfig, LlamaConfig, Qwen2Config
try:
    from transformers import Qwen3Config
except ImportError:
    Qwen3Config = None
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .configuration_intern_vit import InternVisionConfig

logger = logging.get_logger(__name__)


class SaSaSa2VAChatConfig(PretrainedConfig):
    model_type = 'sasasa2va_chat'
    is_composition = True

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
            # SAMURAI
            samurai_mode=False,
            stable_frames_threshold=15,
            stable_ious_threshold=0.3,
            min_obj_score_logits=-1,
            kf_score_weight=0.15,
            memory_bank_iou_threshold=0.5,
            memory_bank_obj_score_threshold=0.0,
            memory_bank_kf_score_threshold=0.0,
            **kwargs
            ):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. Initializing the InternVisionConfig with default values.')

        if llm_config is None:
            llm_config = {}
            logger.info('llm_config is None. Initializing the LlamaConfig config with default values (`LlamaConfig`).')

        self.vision_config = InternVisionConfig(**vision_config)

        if llm_config and 'architectures' in llm_config and llm_config['architectures']:
            if llm_config['architectures'][0] == 'LlamaForCausalLM':
                self.llm_config = LlamaConfig(**llm_config)
            elif llm_config['architectures'][0] == 'InternLM2ForCausalLM':
                self.llm_config = InternLM2Config(**llm_config)
            elif llm_config['architectures'][0] == 'Phi3ForCausalLM':
                self.llm_config = Phi3Config(**llm_config)
            elif llm_config['architectures'][0] == 'Qwen2ForCausalLM':
                self.llm_config = Qwen2Config(**llm_config)
            elif Qwen3Config is not None and llm_config['architectures'][0] == 'Qwen3ForCausalLM':
                self.llm_config = Qwen3Config(**llm_config)
            else:
                raise ValueError('Unsupported architecture: {}'.format(llm_config['architectures'][0]))
        else:
            # 当没有 architectures 或为空时，使用默认 LlamaConfig
            self.llm_config = LlamaConfig()
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
        
        self.samurai_mode = samurai_mode
        self.stable_frames_threshold = stable_frames_threshold
        self.stable_ious_threshold = stable_ious_threshold
        self.min_obj_score_logits = min_obj_score_logits
        self.kf_score_weight = kf_score_weight
        self.memory_bank_iou_threshold = memory_bank_iou_threshold
        self.memory_bank_obj_score_threshold = memory_bank_obj_score_threshold
        self.memory_bank_kf_score_threshold = memory_bank_kf_score_threshold
        
        logger.info(f'samurai_mode: {self.samurai_mode}')
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
        output['samurai_mode'] = self.samurai_mode
        output['stable_frames_threshold'] = self.stable_frames_threshold
        output['stable_ious_threshold'] = self.stable_ious_threshold
        output['min_obj_score_logits'] = self.min_obj_score_logits
        output['kf_score_weight'] = self.kf_score_weight
        output['memory_bank_iou_threshold'] =self.memory_bank_iou_threshold
        output['memory_bank_obj_score_threshold'] = self.memory_bank_obj_score_threshold
        output['memory_bank_kf_score_threshold'] = self.memory_bank_kf_score_threshold

        return output
