import copy

import transformers
from transformers import Qwen2Config
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

logger = logging.get_logger(__name__)

class Sa2VAChatConfigQwen(Qwen2_5_VLConfig):
    model_type = 'sa2va_chat'

    def __init__(
            self,
            template=None,
            **kwargs
        ):

        super().__init__(**kwargs)

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = super().to_dict()

        return output
