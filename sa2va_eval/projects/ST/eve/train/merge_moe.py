# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
import json
import math
import logging
import os
import pathlib
import warnings
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from PIL import Image
from torch.utils.data import Dataset

from eve import conversation as conversation_lib
from eve.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IGNORE_INDEX)
from eve.mm_utils import tokenizer_image_token
from eve.model import *
from eve.train.eve_trainer import EVETrainer


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)
    model_type: Optional[str] = field(default=None,
                                      metadata={"help": "llama3, qwen2"})


@dataclass
class DataArguments:
    is_multimodal: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    vision_tower_lr: Optional[float] = None


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (
        torch.bfloat16 if training_args.bf16 else torch.float32))
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )
    
    if model_args.model_type == 'llama3':
        cfg_pretrained = EVELlamaConfig.from_pretrained(model_args.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = "<|reserved_special_token_0|>"
            tokenizer.pad_token_id = 128002
    elif model_args.model_type == 'qwen2':
        cfg_pretrained = EVEQwen2Config.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError(f"Invalid model_type in model_args: {model_args.model_type}.")

    if model_args.model_type == 'llama3':
        model = EVELlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            config=cfg_pretrained
            )
    elif model_args.model_type == 'qwen2':
        model = EVEQwen2ForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=cfg_pretrained
            )

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    trainer = EVETrainer(model=model,
                         tokenizer=tokenizer,
                         args=training_args)
    
    params_old = dict(model.named_parameters())
    moe_gate_T = model.model.layers[0].mlp.moe_gate_T
    max_num_layer = 0
    for name, param in model.named_parameters():
        if 'moe' in name and 'gate' not in name:
            indexs = name.split('.')[2]
            max_num_layer = max(max_num_layer, int(indexs) + 1)
            if 'down_proj' in name:
                layer_name = f'model.layers.{indexs}.mlp.down_proj.weight'
                gate_name = f'model.layers.{indexs}.mlp.moe_down_gate'
            if 'up_proj' in name:
                layer_name = f'model.layers.{indexs}.mlp.up_proj.weight'
                gate_name = f'model.layers.{indexs}.mlp.moe_up_gate'

            fusied_weight = params_old[layer_name] + params_old[name].t() * torch.tanh(params_old[gate_name] * moe_gate_T)
            params_old[layer_name].data.copy_(fusied_weight)
            del params_old[name]
            del params_old[gate_name]
            # print(name, torch.tanh(params_old[gate_name] * moe_gate_T).item(),
            #       params_old[layer_name].min().item(), params_old[layer_name].max().item(), 
            #       params_old[name].min().item(), params_old[name].max().item())


    for idx in range(max_num_layer):
        del model.model.layers[idx].mlp.moe_down_proj
        del model.model.layers[idx].mlp.moe_down_gate
        del model.model.layers[idx].mlp.moe_up_proj
        del model.model.layers[idx].mlp.moe_up_gate
        
    model.load_state_dict(params_old, strict=True)
    model.to(compute_dtype)

    num_moe_layer = 0
    num_param = 0
    for name, param in model.named_parameters():
        if 'moe' in name:
            num_moe_layer += 1
            print(name)
        else:
            num_param += param.numel()
            print(name, param.dtype)

    print('num_moe_layer', num_moe_layer)
    print('num_param', num_param / 1e9)

    trainer.save_state()

    model.config.use_cache = True
    model.config.tune_MOE = False
    model.config.add_moe = False
    model.config.init_moe = None

    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
