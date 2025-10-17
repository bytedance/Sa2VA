import os
import glob
import torch
import json
import jsonlines
import numpy as np
import pandas as pd

from PIL import Image
import torch.distributed
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing import Optional
from einops import rearrange

from data.templates import PROMPT_TEMPLATE
from data.utils import (
    load_image_to_base64,
    download_image_to_base64,
    load_base64_to_PILImage,
    convert_image_base64_to_patches,
    visualize_patches,
    load_image_bytes_to_base64
)


IGNORE_INDEX = -100
def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def read_jsonlines(file_path):
    with jsonlines.open(file_path) as reader:
        data = [obj for obj in reader]
    return data

def prepare_image_textual_seq(h, w, tokenizer, add_cls=True):
    seq = ""
    tok_len = 0
    
    seq += tokenizer.vis_beg_tok
    tok_len += 1
    for _ in range(h-1):
        seq += tokenizer.vis_patch_tok * w + tokenizer.vis_rsep_tok
        tok_len += (w + 1)
    seq += tokenizer.vis_patch_tok * w + tokenizer.vis_end_tok
    tok_len += (w + 1)
    if add_cls:
        seq += tokenizer.vis_cls_tok
        tok_len += 1
    
    return seq, tok_len

def prepare_image_textual_seq_norowsep(h, w, tokenizer, add_cls=True):
    seq = ""
    tok_len = 0
    
    seq += tokenizer.vis_beg_tok
    tok_len += 1
    
    seq += tokenizer.vis_patch_tok * (w * h)
    tok_len += (w * h)
    
    seq += tokenizer.vis_end_tok
    tok_len += 1
    
    if add_cls:
        seq += tokenizer.vis_cls_tok
        tok_len += 1
    
    return seq, tok_len


def create_single_prefix_mask(prefix_len, max_len):
    attn_mask = torch.zeros(max_len, max_len)
    attn_mask[:prefix_len, :prefix_len] = 1
    causal_mask = torch.tril(torch.ones(max_len, max_len))
    attn_mask = attn_mask.bool() | causal_mask.bool()
    return attn_mask


def generate_mm_pos_ids_singleit(input_ids, vpatch_id, h, w):
    input_ids_pt = torch.Tensor(input_ids).int()
    vpatch_pos = torch.argwhere(input_ids_pt == vpatch_id)
    vpatch_start_pos = vpatch_pos[0].item()
    nt = len(input_ids) - (h*w) + 1
 
    # v_pos
    t_indices = torch.arange(1)
    h_indices = torch.arange(h)
    w_indices = torch.arange(w)
    v_pos_id = torch.stack(torch.meshgrid(t_indices, h_indices, w_indices, indexing='ij'), dim=0)
    v_pos_id = rearrange(v_pos_id, "d t h w -> (t h w) d")  # [h*w, 3]
    v_pos_id += vpatch_start_pos
    position_id = torch.cat(
        [
            torch.arange(vpatch_start_pos).unsqueeze(-1).repeat(1,3),
            v_pos_id,
            torch.arange(nt-vpatch_start_pos-1).unsqueeze(-1).repeat(1,3) + v_pos_id.max() + 1,
        ],
        dim=0
    )
    assert len(input_ids) == position_id.size(0)
    position_id = rearrange(position_id, "slen d -> d slen").long()
    
    return position_id


class SFTModule():
    def prepare_inputs_img(self, images, inputs, tokenizer):
        end_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.pad_token
        bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else ''
        
        NON_VISION_TOKEN = -1
        tokens = []
        vision_patch_indices = []
        vision_patches = []
        labels = []
        
        patches = images
        n_rows, n_cols = patches.shape[:2]
        n_patches = n_rows * n_cols
        patches = patches.view(n_patches, -1)
        
        # ---
        image_text_seq, image_tok_len = prepare_image_textual_seq_norowsep(n_rows, n_cols, tokenizer, getattr(self.config, "add_cls", False))
        # ---
        cur_tokens_pt = tokenizer(image_text_seq, add_special_tokens=False, return_tensors="pt").input_ids[0]
        cur_patch_indices = torch.full_like(cur_tokens_pt, fill_value=NON_VISION_TOKEN)
        assert (cur_tokens_pt==tokenizer.vis_patch_tok_id).sum() == n_patches
        cur_patch_indices[cur_tokens_pt==tokenizer.vis_patch_tok_id] = torch.arange(n_patches)
        
        cur_tokens = cur_tokens_pt.cpu().numpy().tolist()
        cur_patch_indices = cur_patch_indices.cpu().numpy().tolist()
        assert len(cur_tokens) == len(cur_patch_indices), f"{len(cur_tokens)} != {len(cur_patch_indices)}"
        
        tokens.extend(cur_tokens)
        labels.extend([-100] * len(cur_tokens)) 
        vision_patch_indices.extend(cur_patch_indices)
        vision_patches.extend(patches.numpy().astype(np.float16))
        
        
        for idx, i in enumerate(inputs):
            if idx % 2 == 0:
                # system/user part
                if idx == 0:
                    i = i.replace("<image>\n", '').replace("\n<image>", '')
                    c_new = bos_token + self.template['INSTRUCTION'].format(input=i.strip())                                   
                else:
                    c_new = self.template['INSTRUCTION'].format(input=i.strip())
                _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
                cur_tokens = _tokenized["input_ids"].squeeze(0)
                tokens.extend(cur_tokens)
                labels.extend([-100] * len(cur_tokens))
                vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
            else:
                # assistant part
                i = i + end_token
                _tokenized = tokenizer(i, return_tensors="pt", add_special_tokens=False)
                cur_tokens = _tokenized["input_ids"].squeeze(0)
                tokens.extend(cur_tokens)
                labels.extend(cur_tokens)
                vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
        
        position_ids = generate_mm_pos_ids_singleit(tokens, tokenizer.vis_patch_tok_id, n_rows, n_cols)  # [3, slen]
        attention_masks = create_single_prefix_mask(image_tok_len, len(tokens)).unsqueeze(0) # [1, slen, slen]

        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        if len(vision_patches) > 0:
            # convert vision patches to numpy
            vision_patches = np.array(vision_patches)
            vision_patches = torch.Tensor(vision_patches).bfloat16()
        else:
            vision_patches = None
        

        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            position_ids = position_ids[:, :self.max_position_embeddings]
            attention_masks = attention_masks[:, :self.max_position_embeddings, :self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
            vision_patches = vision_patches[:self.max_position_embeddings]

        
        return tokens,  position_ids, attention_masks, vision_patches, vision_patch_indices, labels



    def prepare_inputs(self, inputs, flag, tokenizer):
        end_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.pad_token
        bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else ''

        NON_VISION_TOKEN = -1
        tokens = []
        attention_masks = []
        vision_patch_indices = []
        vision_patches = []
        labels = []

        for idx, i in enumerate(inputs):
            if idx % 2 == 0:
                if idx == 0:
                    c_new = bos_token + self.template['INSTRUCTION'][0].format(input=i.strip())                                   
                else:
                    c_new = self.template['INSTRUCTION'][1].format(input=i.strip())
                _tokenized = tokenizer(c_new, return_tensors="pt", add_special_tokens=False)
                cur_tokens = _tokenized["input_ids"].squeeze(0)
                cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
                tokens.extend(cur_tokens)
                labels.extend([-100] * len(cur_tokens))
                attention_masks.extend(cur_attention_mask)
                vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
            else:
                i = i + end_token
                _tokenized = tokenizer(i, return_tensors="pt", add_special_tokens=False)
                cur_tokens = _tokenized["input_ids"].squeeze(0)
                cur_attention_mask = _tokenized["attention_mask"].squeeze(0)
                tokens.extend(cur_tokens)
                labels.extend(cur_tokens)
                attention_masks.extend(cur_attention_mask)
                vision_patch_indices.extend([NON_VISION_TOKEN] * len(cur_tokens))
        
        
        if len(tokens) > self.max_position_embeddings:
            tokens = tokens[:self.max_position_embeddings]
            labels = labels[:self.max_position_embeddings]
            attention_masks = attention_masks[:self.max_position_embeddings]
            vision_patch_indices = vision_patch_indices[:self.max_position_embeddings]
        
        tokens = torch.Tensor(tokens).long()
        labels = torch.Tensor(labels).long()
        attention_masks = torch.Tensor(attention_masks).long()
        vision_patches = None
        vision_patch_indices = torch.Tensor(vision_patch_indices).long()
        position_ids = torch.arange(len(tokens)).unsqueeze(0).expand(3,-1).clone().long()
        
        return tokens, position_ids, attention_masks, vision_patches, vision_patch_indices, labels
    
    def collate_fn(self, batch):
        try:
            assert len(batch) == 1
            for i, tgt_item in enumerate(batch):
                conversation_li = []
                conversations = tgt_item['conversations']
                for item in conversations:
                    if type(item) is str:
                        conversation_li.append(item)
                    else:
                        conversation_li.append(item['value'])
                if 'image' in tgt_item:
                    # orig_img_path = os.path.join(self.img_dir, tgt_item['image'])
                    orig_img_path = tgt_item['image']
                    img_base64 = load_image_to_base64(orig_img_path)
                    img_patches = convert_image_base64_to_patches(img_base64, patch_size=self.patch_size,fix_resolution=self.fix_resolution)
                    tokens, position_ids, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs_img(img_patches, conversation_li, self.tokenizer)
                else:
                    tokens, position_ids, attention_masks, vision_patches, vision_patch_indices, labels = self.prepare_inputs(conversation_li, 0, self.tokenizer)

            return {
                "input_ids": tokens.unsqueeze(0),
                "position_ids": position_ids.unsqueeze(1),
                "attention_mask": attention_masks.unsqueeze(0),
                "vision_patches": vision_patches,
                "vision_patch_indices": vision_patch_indices.unsqueeze(0),
                "labels": labels.unsqueeze(0)
            }
                
        except Exception as e:
            print(e)
            return None
    
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.all_data,
            batch_size=self.config["data"]["batch_size"],
            shuffle=True,
            num_workers=self.config["data"]["num_workers"],
            collate_fn=self.collate_fn,
            pin_memory=True
        )
        
    
    def __init__(self, config: dict, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.max_position_embeddings = config["data"]["max_position_embeddings"]
        self.img_dir = config["data"]["img_dir"]
        template_name = config["data"].get("template", "mistral")
        self.template = PROMPT_TEMPLATE[template_name]
        ann_file = config["data"]["train_data"]
        all_data = []
        ann_files = ann_file
        img_dirs = self.img_dir
        for sub_ann_file, sub_img_dir in zip(ann_files,img_dirs):
            if sub_ann_file.endswith(".jsonl"):
                data_dict = read_jsonlines(sub_ann_file)
            elif sub_ann_file.endswith(".json"):
                data_dict = json.load(open(sub_ann_file))
            for i in range(len(data_dict)):
                if 'image' in data_dict[i]:
                    data_dict[i]['image'] = os.path.join(sub_img_dir, data_dict[i]['image'])
            all_data.extend(data_dict)
        self.all_data = all_data
        self.patch_size = config["data"].get("patch_size", 32)
        self.fix_resolution = config["data"].get("fix_resolution", False)


def get_data(config, tokenizer):
    return SFTModule(config, tokenizer)

