from typing import Literal
from collections import OrderedDict
from pycocotools import mask as _mask
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModel
from xtuner.registry import BUILDER
from xtuner.model.utils import guess_load_checkpoint

from third_parts.mmdet.models.utils.point_sample import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness

from projects.sa2va.models import Sa2VAModel

class SaSaSa2VAModel(Sa2VAModel):
    def __init__(self,
                 mllm,
                 tokenizer,
                 grounding_encoder,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 frozen_sam2_decoder=True,
                 special_tokens=None,
                 loss_sample_points=False,
                 num_points=12544,
                 template=None,
                 # for arch selection
                 arch_type:Literal['intern_vl', 'qwen', 'llava']='intern_vl',
                 # ext
                 preprocessor=None,
                 # bs
                 training_bs:int=0,
                 ):
        super(SaSaSa2VAModel, self).__init__(
            mllm=mllm, 
            tokenizer=tokenizer, 
            grounding_encoder=grounding_encoder, 
            loss_mask=loss_mask, 
            loss_dice=loss_dice, 
            torch_dtype=torch_dtype, 
            pretrained_pth=pretrained_pth, 
            frozen_sam2_decoder=frozen_sam2_decoder, 
            special_tokens=special_tokens, 
            loss_sample_points=loss_sample_points, 
            num_points=num_points, 
            template=template, 
            arch_type=arch_type, 
            preprocessor=preprocessor, 
            training_bs=training_bs,
        )

    def forward(self, data, data_samples=None, mode='loss'):
        g_pixel_values = data.pop('g_pixel_values', None)
        gt_masks = data.pop('masks', None)
        frames_per_batch = data.pop('frames_per_batch', None)
        input_ids = data['input_ids']
        output = self.mllm(data, data_samples, mode)

        if gt_masks is None:
             # require zero seg datas
            seg_valid = False
            g_pixel_values, frames_per_batch, gt_masks = self._get_pesudo_data(
                dtype=self.torch_dtype,
                device=input_ids.device,
            )
        else:
            seg_valid = True

        ori_size_list = []
        for i_bs, mask in enumerate(gt_masks):
            mask_shape = mask.shape[-2:]
            ori_size_list += [mask_shape] * frames_per_batch[i_bs]

        seg_token_mask = input_ids == self.seg_token_idx    # [B, N]

        hidden_states = output.hidden_states
        hidden_states = self.text_hidden_fcs(hidden_states[-1])  # [B, N, C] 
        _zero = hidden_states.mean() * 0.0
        if seg_valid:
            pred_embeddings = hidden_states[seg_token_mask] + _zero 
        else:
            pred_embeddings = hidden_states[:, :5].flatten(0, 1) + _zero

        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 5

        ## pred_embeddings -> [Num_seg_token, C]
        pred_embeddings_list_ = torch.split(pred_embeddings, seg_token_counts.tolist(), dim=0)  
        pred_embeddings_list = []
        for item in pred_embeddings_list_:
            if len(item) != 0:
                pred_embeddings_list.append(item)
     
        if not seg_valid:
            pred_embeddings_list_video = self.genetate_video_pred_embeddings(
                pred_embeddings_list, frames_per_batch)
        else:
            pred_embeddings_list_video = self.generate_video_pred_embeddings_sasasa2va(
                pred_embeddings_list, frames_per_batch)

        gt_masks_video = self.process_video_gt_masks(gt_masks, frames_per_batch) 
        pred_embeddings_list_video, gt_masks_video = self.check_obj_number(
            pred_embeddings_list_video, gt_masks_video
        )
        g_pixel_values = torch.stack([
            self.grounding_encoder.preprocess_image(pixel) for pixel in g_pixel_values
        ])
        num_objs = pred_embeddings_list_video[0].shape[0]
        num_frames = len(pred_embeddings_list_video) 
        language_embeddings = torch.cat(pred_embeddings_list_video, dim=0)[:, None] 
        assert num_objs == 5
        assert num_objs == gt_masks_video[0].shape[0], "Num of objects in gt_masks and pred_embeddings are not equal !!!"
        assert num_frames == len(gt_masks_video), "Num of frames in gt_masks and frames_per_batch are not equal !!!"
        sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values, expand_size=num_objs)
        pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))
        gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_masks[0].shape[-2:], mode='nearest').squeeze(0) for gt_mask in gt_masks_video]
        gt_masks = torch.cat(gt_masks, dim=0)
        pred_masks = pred_masks.flatten(0, 1) 

        bs = len(pred_masks)
        loss_mask, loss_dice = 0, 0
        if len(pred_masks) != len(gt_masks):
            # drop this data
            print(f"Pred mask shape {pred_masks.shape} is not equal to gt_mask shape {gt_masks.shape} !!!")
            min_num = min(len(pred_masks), len(gt_masks))
            pred_masks = pred_masks[:min_num]
            gt_masks = gt_masks[:min_num]
            seg_valid = False

        if self.loss_sample_points:
            sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(
                sampled_pred_mask,
                sampled_gt_mask, avg_factor=(len(gt_masks) + 1e-4))
            sam_loss_mask = self.loss_mask(
                sampled_pred_mask.reshape(-1),
                sampled_gt_mask.reshape(-1),
                avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
        else:
            sam_loss_mask = self.loss_mask(pred_masks, gt_masks)
            sam_loss_dice = self.loss_dice(pred_masks, gt_masks)
        loss_mask += sam_loss_mask
        loss_dice += sam_loss_dice

        if not seg_valid:
            _scale = 0.0
        else:
            _scale = 1.0
        loss_mask = loss_mask * _scale
        loss_dice = loss_dice * _scale

        loss_dict = {
            'loss_mask': loss_mask,
            'loss_dice': loss_dice,
            'llm_loss': output.loss,
        }
        return loss_dict

    def generate_video_pred_embeddings_sasasa2va(self, pred_embeddings_list, frames_per_batch):
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            if pred_embedding_batch.shape[0] % frame_nums == 0:
                reshaped_pred_embedding_batch = pred_embedding_batch.reshape(-1, frame_nums, pred_embedding_batch.shape[-1])
                pred_embeddings_list_video += [reshaped_pred_embedding_batch[:, i, :] for i in range(frame_nums)]
            else:
                raise ValueError("pred_embedding_batch.shape[0]:{} is not divisible by frame_nums:{}".format(pred_embedding_batch.shape[0], frame_nums))
        return pred_embeddings_list_video


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def mask_to_rle(mask):
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle
