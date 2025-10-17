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
    version: Optional[str] = field(default=None,
                                   metadata={"help": "plain, llama3, qwen2"})
    vision_tower: Optional[str] = field(default=None)
    vision_tower_hidden_size: int = field(default=1024)
    add_moe: bool = field(default=False)
    moe_part: Optional[str] = field(default=None, 
                                    metadata={"help": "self_attn-mlp, layernorm-self_attn-mlp"})
    tune_LLM: bool = field(default=False)
    tune_VE: bool = field(default=False)
    tune_MOE: bool = field(default=False)
    

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    json_path: str = field(default=None,
                           metadata={"help": "Path to the split json data."})
    presave_lengths: str = field(default=None,
                           metadata={"help": "Path to the json length data."})
    presave_modality_lengths: str = field(default=None,
                           metadata={"help": "Path to the json modality length data."})                                                      
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


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


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(
                    DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + \
                    '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

    return sources


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:

    conv = conversation_lib.default_conversation.copy()
    assert conv.sep_style == conversation_lib.SeparatorStyle.PLAIN

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        if has_image:
            source[0]['value'] = DEFAULT_IMAGE_TOKEN
            conversation = DEFAULT_IMAGE_TOKEN + source[1]['value'].strip() + \
                           conversation_lib.default_conversation.sep
        else:
            conversation = source[1]['value'].strip() + \
                           conversation_lib.default_conversation.sep
        conversations.append(conversation)

    # tokenize conversations
    if has_image:
        input_ids = [
            tokenizer_image_token(prompt, tokenizer, return_tensors='pt') 
            for prompt in conversations]
    else:
        input_ids = [tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0] for prompt in conversations] 
    
    targets = copy.deepcopy(input_ids)
    if has_image:
        for target, source in zip(targets, sources):
            tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
            target[:tokenized_len] = IGNORE_INDEX
    
    return dict(input_ids=input_ids, labels=targets)


def preprocess_eve(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') 
            for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    
    offset = 0 if (input_ids[0][0] != tokenizer.bos_token_id) else 1
    # Mask targets
    sep = conv.sep + conv.roles[1] + ":"
    # Llama3 tokenizer has the token for whitespace
    # Typically, the token after whitespace will be naturally encoded as one token with whitespace
    # some special cases like ": 3" will be encoded as :, whitespace, 3; 3 tokens. Only in this case, the loss on whitespace will be calculated
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = offset
        target[:cur_len] = IGNORE_INDEX
        
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids)

            target[cur_len : cur_len + instruction_len - offset] = IGNORE_INDEX
            cur_len += round_len - offset + 1 #starting from index 0, then cur_len will not cover eos token

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer, has_image=has_image)
    else:
        return preprocess_eve(sources, tokenizer, has_image=has_image)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        self.list_data_dict = json.load(open(data_path, "r"))
        rank0_print(f"Formatting {len(self.list_data_dict)} inputs...Skip in lazy mode")

        self.data_is_index = type(self.list_data_dict[0]) == str
        self.json_path = data_args.json_path
        self.presave_lengths = data_args.presave_lengths
        self.presave_modality_lengths = data_args.presave_modality_lengths

        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        if self.presave_lengths is None:
            length_list = []
            for sample in self.list_data_dict:
                if self.data_is_index:
                    sample = json.load(open(os.path.join(self.json_path, f'{sample}'), 'r'))
                img_tokens = 2500 if 'image' in sample else 0
                length_list.append(sum(len(conv['value'].split())
                                for conv in sample['conversations']) + img_tokens)
        else:
            length_list = json.load(open(self.presave_lengths, 'r'))
        return length_list

    @property
    def modality_lengths(self):
        if self.presave_modality_lengths is None:
            length_list = []
            for sample in self.list_data_dict:
                if self.data_is_index:
                    sample = json.load(open(os.path.join(self.json_path, f'{sample}'), 'r'))
                cur_len = sum(len(conv['value'].split())
                            for conv in sample['conversations'])
                cur_len = cur_len if 'image' in sample else -cur_len
                length_list.append(cur_len)
        else:
            length_list = json.load(open(self.presave_modality_lengths, 'r'))
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if self.data_is_index:
            sources = json.load(open(os.path.join(self.json_path, f'{sources}'), 'r'))
        sources_has_image = 'image' in sources

        if isinstance(i, int):
            sources = [sources]
        assert len(
            sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if sources_has_image:
            image_file = sources[0]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            raw_image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            
            width, height = raw_image.size
            scale_ratio = math.sqrt(width * height) / processor.max_size

            if scale_ratio > 1:
                width = width / scale_ratio
                height = height / scale_ratio

            min_edge = processor.patch_stride * processor.dense_stride
            width = max(int(round(width / min_edge)), 1) * min_edge
            height = max(int(round(height / min_edge)), 1) * min_edge

            new_image = raw_image.resize((width, height))
            image = processor.preprocess(new_image, return_tensors='pt')['pixel_values'][0]
            del raw_image
            del new_image

            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=sources_has_image)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if sources_has_image:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            processor = self.data_args.image_processor
            min_edge = processor.patch_stride * processor.dense_stride
            background_color = tuple(int(x*255) for x in processor.image_mean)

            image = Image.new('RGB', (min_edge, min_edge), background_color)
            data_dict['image'] = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                          data_path=data_args.data_path,
                                          data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


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

    if hasattr(cfg_pretrained, "mm_vision_tower"):
        cfg_pretrained.mm_vision_tower = model_args.vision_tower

    cfg_pretrained.pad_token_id = tokenizer.pad_token_id
    cfg_pretrained.add_moe = model_args.add_moe
    cfg_pretrained.moe_part = model_args.moe_part

    assert model_args.vision_tower is not None
    assert tokenizer.eos_token_id and tokenizer.pad_token_id
    assert tokenizer.eos_token_id != tokenizer.pad_token_id

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
    
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        raise ValueError(f"Invalid version in model_args: {model_args.version}.")

    if model_args.vision_tower is not None:  # Start the vision encoder initialization
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        data_args.image_processor = vision_tower.image_processor
        
        for name, param in model.named_parameters():
            param.requires_grad = False
            if 'vision_tower' in name:
                if model_args.tune_VE:
                    param.requires_grad = True
            elif 'moe' in name:
                if model_args.tune_MOE:
                    param.requires_grad = True
            else:
                if model_args.tune_LLM:
                    param.requires_grad = True

        data_args.is_multimodal = True
        model.config.vision_tower_lr = training_args.vision_tower_lr
        model.config.tune_VE = training_args.tune_VE = model_args.tune_VE
        model.config.tune_LLM = training_args.tune_LLM = model_args.tune_LLM
        model.config.tune_MOE = training_args.tune_MOE = model_args.tune_MOE
    
    num_train_param, num_fixed_param = 0, 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_train_param += param.numel()
        else:
            num_fixed_param += param.numel()

    rank0_print('Number of trainable parameters: %.2f B' % (num_train_param/ 1e9))
    rank0_print('Number of frozen parameters: %.2f B' % (num_fixed_param/ 1e9))
    
    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = EVETrainer(model=model,
                         tokenizer=tokenizer,
                         args=training_args,
                         **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
