import os
import warnings

import torch
from transformers import AutoTokenizer
                           
from projects.ST.eve.model import *


def load_pretrained_model(model_path, model_type, device_map="auto", device="cuda"):
    if model_type not in ['qwen2', 'llama3']:
        raise ValueError(f"Invalid Model Type {model_type}")

    kwargs = {"device_map": device_map}
    kwargs['torch_dtype'] = torch.float16

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    if model_type == 'llama3':
        model = EVELlamaForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
            )
    elif model_type == 'qwen2':
        model = EVEQwen2ForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
            )
    else:
        raise ValueError(f"Invalid model_type in args: {model_type}.")

    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 4096

    if model_type == 'llama3' and tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = 128002
        tokenizer.pad_token = "<|reserved_special_token_0|>"
    
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return tokenizer, model, image_processor, context_len
