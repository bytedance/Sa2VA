from typing import Dict, Sequence

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from xtuner.parallel.sequence import (get_sequence_parallel_world_size,
                                      pad_for_sequence_parallel)
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX, IGNORE_INDEX
from einops import rearrange

NON_VISION_TOKEN = -1

def generate_mm_pos_ids_singleit(input_ids, vpatch_id, h, w):
    if h * w == 0:
        nt = len(input_ids)
        # pure text
        position_id = torch.arange(nt).unsqueeze(-1).repeat(1, 3)
        assert len(input_ids) == position_id.size(0)
        position_id = rearrange(position_id, "slen d -> d slen").long()
        return position_id
    input_ids_pt = torch.Tensor(input_ids).int()
    vpatch_pos = torch.argwhere(input_ids_pt == vpatch_id)
    vpatch_start_pos = vpatch_pos[0].item()
    nt = len(input_ids) - (h * w) + 1

    # v_pos
    t_indices = torch.arange(1)
    h_indices = torch.arange(h)
    w_indices = torch.arange(w)
    v_pos_id = torch.stack(torch.meshgrid(t_indices, h_indices, w_indices, indexing='ij'), dim=0)
    v_pos_id = rearrange(v_pos_id, "d t h w -> (t h w) d")  # [h*w, 3]
    v_pos_id += vpatch_start_pos
    position_id = torch.cat(
        [
            torch.arange(vpatch_start_pos).unsqueeze(-1).repeat(1, 3),
            v_pos_id,
            torch.arange(nt - vpatch_start_pos - 1).unsqueeze(-1).repeat(1, 3) + v_pos_id.max() + 1,
        ],
        dim=0
    )
    assert len(input_ids) == position_id.size(0)
    position_id = rearrange(position_id, "slen d -> d slen").long()

    return position_id

def st_collate_fn(instances: Sequence[Dict],
                  pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
                  return_hf_format: bool = False,
                  use_varlen_attn: bool = False):
    seq_parallel_world_size = get_sequence_parallel_world_size()

    vision_patch_idx = instances[0].get('vision_patch_idx')

    input_ids, labels = [], []
    has_image = any(inst.get('vision_patches') is not None for inst in instances)
    has_mask = any(inst.get('masks') is not None for inst in instances)

    if use_varlen_attn:
        position_ids, cumulative_len = [], []
        assert len(instances) == 1, (
            f'If utilizing varlen attention, the batch size should be'
            f' set to 1, but got {len(instances)}')
        assert not has_image, 'Currently, it is not configured to '
        'accommodate the use of varlen Attention in multimodal training'

    patch_nums_per_images = []
    vision_start_end = []
    vision_patch_indices = []
    vp_token_indices = []
    vp_feat_indices = []

    if has_image:
        vision_patches = []
    else:
        vision_patches = None
    if has_mask:
        masks = []
    else:
        masks = None

    _vision_indexes_prefix = 0
    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))
        patch_nums_per_images.append(example['patch_nums_per_images'])
        vision_start_end.append(example['vision_start_end'])

        # compute new multi-batch vision patch indices
        batch_vision_patch_indices = torch.LongTensor(example['vision_patch_indices'])
        if 'vp_token_indices' in example.keys():
            batch_vp_token_indices = torch.LongTensor(example['vp_token_indices'])
        else:
            batch_vp_token_indices = torch.LongTensor(example['vision_patch_indices']) * 0 + NON_VISION_TOKEN
            batch_vp_token_indices[batch_vision_patch_indices != -1] = -2
        if 'vp_feat_indices' in example.keys():
            batch_vp_feat_indices = torch.LongTensor(example['vp_feat_indices'])
        else:
            batch_vp_feat_indices = torch.LongTensor(example['vision_patch_indices']) * 0 + NON_VISION_TOKEN

        batch_vision_patch_indices[batch_vision_patch_indices!=NON_VISION_TOKEN] += _vision_indexes_prefix
        batch_vp_feat_indices[batch_vp_feat_indices!=NON_VISION_TOKEN] += _vision_indexes_prefix
        _vision_indexes_prefix = max(torch.max(batch_vision_patch_indices), 0)
        vision_patch_indices.append(batch_vision_patch_indices)
        vp_feat_indices.append(batch_vp_feat_indices)
        vp_token_indices.append(batch_vp_token_indices)

        if use_varlen_attn:
            cumulative_len.append(torch.IntTensor(example['cumulative_len']))
            position_ids.append(torch.LongTensor(example['position_ids']))

        if has_image:
            if 'vision_patches' in example.keys():
                vision_patches.append(example['vision_patches'])
        if has_mask:
            if 'masks' in example.keys() and example['masks'] is not None:
                masks.append(example['masks'])
            else:
                masks.append(None)

    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        vision_patch_indices = pad_sequence(
            vision_patch_indices, batch_first=True, padding_value=NON_VISION_TOKEN)
        vp_feat_indices = pad_sequence(
            vp_feat_indices, batch_first=True, padding_value=NON_VISION_TOKEN)
        vp_token_indices = pad_sequence(
            vp_token_indices, batch_first=True, padding_value=NON_VISION_TOKEN)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        vision_patch_indices = torch.stack(vision_patch_indices)
        vp_feat_indices = torch.stack(vp_feat_indices)
        vp_token_indices = torch.stack(vp_token_indices)

    if use_varlen_attn:
        assert input_ids.size(1) % seq_parallel_world_size == 0
        attention_mask = None
        position_ids = torch.stack(position_ids, dim=0)
    else:
        # Some tokenizers have the same eos token and pad token, so input_ids
        # cannot be masked directly based on the pad token id.
        attention_mask = torch.zeros(input_ids.shape[0], 1, input_ids.shape[1], input_ids.shape[1]).bool()
        for i, length in enumerate(ori_length):
            attention_mask[i, 0, :length, :length] = create_single_prefix_mask(vision_start_end[i], length)

        bs, seq_len = input_ids.shape

        position_ids = []
        for input_id, patch_nums_per_image in zip(input_ids, patch_nums_per_images):
            position_id = generate_mm_pos_ids_singleit(
                input_id.cpu().numpy().tolist(), vision_patch_idx,
                patch_nums_per_image[0], patch_nums_per_image[1])
            position_ids.append(position_id)
        position_ids = torch.stack(position_ids, dim=1)

    if seq_parallel_world_size > 1:
        input_ids = pad_for_sequence_parallel(input_ids, pad_index)
        labels = pad_for_sequence_parallel(labels, IGNORE_INDEX)
        position_ids = pad_for_sequence_parallel(position_ids, 0)
        if attention_mask is not None:
            attention_mask = pad_for_sequence_parallel(attention_mask, 0)

    if has_image:
        if len(vision_patches) == 0:
            vision_patches = None
        else:
            vision_patches = torch.cat(vision_patches, dim=0)

    if use_varlen_attn:
        max_seqlen = (
            cumulative_len[0][1:] -  # noqa: W504
            cumulative_len[0][:-1]).max().item()
        data_dict = {
            'input_ids': input_ids,
            'cumulative_len': cumulative_len,
            'position_ids': position_ids,
            'labels': labels,
            'max_seqlen': max_seqlen,
            'vision_patch_indices': vision_patch_indices,
            'vp_feat_indices': vp_feat_indices,
            'vp_token_indices': vp_token_indices,
            'masks': masks,
            'vision_patches': vision_patches,
            'patch_nums_per_images': patch_nums_per_images
        }
    else:
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'labels': labels,
            'vision_patch_indices': vision_patch_indices,
            'vp_feat_indices': vp_feat_indices,
            'vp_token_indices': vp_token_indices,
            'masks': masks,
            'vision_patches': vision_patches,
            'patch_nums_per_images': patch_nums_per_images
        }

    if return_hf_format:
        return data_dict
    else:
        return {'data': data_dict, 'data_samples': None}

def create_single_prefix_mask(vision_start_end, max_len):
    if vision_start_end is None:
        # pure text
        attn_mask = torch.tril(torch.ones(max_len, max_len))
    else:
        attn_mask = torch.zeros(max_len, max_len)
        attn_mask[vision_start_end[0]-1:vision_start_end[1]+1, vision_start_end[0]-1:vision_start_end[1]+1] = 1
        causal_mask = torch.tril(torch.ones(max_len, max_len))
        attn_mask = attn_mask.bool() | causal_mask.bool()
    return attn_mask