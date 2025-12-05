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

from peft import PeftModelForCausalLM

from transformers import AutoImageProcessor, AutoVideoProcessor, GenerationConfig


# import modules for grpo
from scipy.optimize import linear_sum_assignment
import re
import os
import copy
import json
from xtuner.utils import DEFAULT_PAD_TOKEN_INDEX

class MaskMatcher:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def compute_iou_matrix(self, masks_a: torch.Tensor, masks_b: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of binary masks.
        Args:
            masks_a: Tensor of shape [N, H, W]
            masks_b: Tensor of shape [M, H, W]
        Returns:
            iou_matrix: Tensor of shape [N, M]
        """
        N, H, W = masks_a.shape
        M = masks_b.shape[0]

        masks_a = masks_a.reshape(N, -1).float()  # [N, H*W]
        masks_b = masks_b.reshape(M, -1).float()  # [M, H*W]

        # Intersection
        intersection = torch.matmul(masks_a, masks_b.T)  # [N, M]

        # Union
        area_a = masks_a.sum(dim=1).unsqueeze(1)  # [N, 1]
        area_b = masks_b.sum(dim=1).unsqueeze(0)  # [1, M]
        union = area_a + area_b - intersection

        iou_matrix = intersection / (union + self.eps)
        return iou_matrix

    def match(self, masks_a: torch.Tensor, masks_b: torch.Tensor):
        """
        Perform bipartite matching to maximize total IoU.
        Args:
            masks_a: Tensor of shape [N, H, W]
            masks_b: Tensor of shape [M, H, W]
        Returns:
            matched_pairs: List of tuples (i, j), where i is index in A, j in B
            iou_matrix: Tensor of shape [N, M]
        """
        iou_matrix = self.compute_iou_matrix(masks_a, masks_b)
        if len(masks_a) == 1 or len(masks_b) == 1:
            return [(0, 0)], iou_matrix

        cost_matrix = 1 - iou_matrix.cpu().numpy()
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
        return matched_pairs, iou_matrix
    

def format_rewards_v1(predictions_texts):
    pattern = r"<think>.*?</think>.*?<answer>.*?[SEG].*?</answer>.*?"
    completion_contents = predictions_texts
    matches = [re.fullmatch(pattern, content, re.DOTALL) and "reasoning process here" not in content for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def format_rewards_v2(predictions_texts):
    pattern_full = r"<think>.*?\[SEG\].*?</think>.*?<answer>.*?\[SEG\].*?</answer>.*?"
    pattern_answer = r"<think>.*?</think>.*?<answer>.*?\[SEG\].*?</answer>.*?"
    rewards = []
    for content in predictions_texts:
        match = re.fullmatch(pattern_full, content, re.DOTALL)
        if match:
            rewards.append(1.0)
        elif re.fullmatch(pattern_answer, content, re.DOTALL):
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def answer_rewards_v3(predictions_masks, gt_masks, answer_idxs=None):
    gt_masks = torch.nn.functional.interpolate(
        gt_masks.unsqueeze(1), size=(256, 256), mode='nearest',
    ).squeeze(1)
    lam = 0.1
    rewards = []
    for idx, pred_mask in enumerate(predictions_masks):
        gt_mask = gt_masks
        if pred_mask is None:
            rewards.append(0.0)
        else:
            pred_mask = pred_mask.to(torch.float32)
            if answer_idxs is not None:
                if len(answer_idxs[idx]) == 0:
                    rewards.append(0.0)
                    continue
                pred_mask = pred_mask[answer_idxs[idx]].sigmoid() > 0.5

            matcher = MaskMatcher()
            matched_pairs, iou_matrix = matcher.match(pred_mask, gt_mask)
            N, M = pred_mask.shape[0], gt_mask.shape[0]
            
            iou_list = []
            for pair in matched_pairs:
                if iou_matrix[pair[0], pair[1]] > 0.5:
                    iou_list.append(iou_matrix[pair[0], pair[1]])
            if len(iou_list) == 0:
                rewards.append(0.0)
                continue
            iou = torch.mean(torch.stack(iou_list)) - lam * (N + M - 2 * (len(iou_list)))
            rewards.append(iou.cpu().item())
    return rewards



def find_seg_indices(text):
    all_seg_indices = [m.start() for m in re.finditer(r'\[SEG\]', text)]
    answer_spans = [(m.start(), m.end()) for m in re.finditer(r'<answer>.*?</answer>', text, re.DOTALL)]
    if len(answer_spans) == 0:
        return [], []
    if len(answer_spans) > 1:
        return [], []
        # raise ValueError(f"There should be only one <answer> tag in the text. {text}")
    answer_span = answer_spans[0]
    start, end = answer_span
    
    seg_indices_in_reason = []
    seg_indices_in_answer = []
    for idx, seg_ind in enumerate(all_seg_indices):
        if start <= seg_ind < end:
            seg_indices_in_answer.append(idx)
        elif seg_ind < start:
            seg_indices_in_reason.append(idx)
        else:
            continue
    return seg_indices_in_reason, seg_indices_in_answer


class Sa2VAModelGRPOv2(BaseModel):
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
                 # preprocessor=None,
                 # bs
                 training_bs:int=0,
                 train_segmentation_grpo=False,
                 ):
        super().__init__()
        if special_tokens is None:
            special_tokens = ['[SEG]']

        self.mllm = BUILDER.build(mllm)
        self.arch_type = arch_type

        tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens(tokenizer, special_tokens)

        if arch_type == 'qwen':
            image_processor = AutoImageProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            video_processor = AutoVideoProcessor.from_pretrained(mllm['model_path'], trust_remote_code=True)
            self.mllm._init_processor(image_processor, video_processor)

        self.grounding_encoder = BUILDER.build(grounding_encoder)
        self.grounding_encoder.requires_grad_(False)
        if not frozen_sam2_decoder:
            self.grounding_encoder.sam2_model.sam_mask_decoder.requires_grad_(True)

        # FIX BUG: Untie weights for Qwen model
        if self.arch_type == 'qwen' and self.mllm.model.config.tie_word_embeddings:
            print("Untying embed_tokens and lm_head weights for Qwen model.")
            self.mllm.model.config.tie_word_embeddings = False
            lm_head = self.mllm.model.get_output_embeddings()
            if lm_head is not None:
                input_embeddings = self.mllm.model.get_input_embeddings()
                lm_head.weight = nn.Parameter(input_embeddings.weight.clone())

        in_dim = self.mllm.get_embedding_size()
        out_dim = self.grounding_encoder.hidden_dim
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )
        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            print("-="*40)
            # print("pretrained_state_dict keys:", pretrained_state_dict.keys())
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

            # FIX BUG: Force update lm_head weight after loading state_dict
            if self.arch_type == 'qwen':
                print("Force updating lm_head weight from pretrained state_dict.")
                lm_head_key = 'mllm.model.lm_head.weight'
                if lm_head_key in pretrained_state_dict:
                    lm_head_weight = pretrained_state_dict[lm_head_key]
                    self.mllm.model.get_output_embeddings().weight.data.copy_(lm_head_weight)
                    print(f"Successfully updated lm_head weight from key: {lm_head_key}")
                else:
                    print(f"Warning: lm_head weight key '{lm_head_key}' not found in pretrained_state_dict.")

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.template = template
        self.bs = training_bs

        if self.mllm.use_llm_lora:
            self.mllm.manual_prepare_llm_for_lora()

        # Print gradient status of all weights in self.mllm.model.base_model.model
        print("\n" + "="*80)
        print("GRADIENT STATUS OF MLLM.MODEL WEIGHTS")
        print("="*80)
        
        try:
            base_model = self.mllm.model
            total_params = 0
            trainable_params = 0
            
            for name, param in base_model.named_parameters():
                total_params += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
                    grad_status = "✓ TRAINABLE"
                else:
                    grad_status = "✗ FROZEN"
                
                print(f"{name:<60} | {grad_status} | Shape: {tuple(param.shape)} | Params: {param.numel():,}")
            
            print("-" * 80)
            print(f"SUMMARY:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Frozen parameters: {total_params - trainable_params:,}")
            print(f"  Trainable ratio: {trainable_params/total_params*100:.2f}%")
            print("=" * 80)
            
        except Exception as e:
            print(f"Failed to access self.mllm.model: {e}")
            print("Available attributes in self.mllm.model:")
            print([attr for attr in dir(self.mllm.model) if not attr.startswith('_')])



        # ---- preparing for the GRPO ----
        self.tokenizer = self.mllm.tokenizer
        self.generation_nums = 4
        self.train_segmentation = train_segmentation_grpo

        default_generation_kwargs = dict(
            max_new_tokens=1024,
            do_sample=True,
            temperature=1.2,
            top_p=0.9,
            top_k=0,
            num_return_sequences=self.generation_nums,
        )
        self.gen_config = GenerationConfig(**default_generation_kwargs)

        self.format_reward_weight = 1.0
        self.answer_reward_weight = 1.0

        self.seg_ratio = 1.0
        self.beta = 0.04

        # Visualization settings
        self.visualization_step_interval = 0
        self.visualization_dir = "work_dirs/grpo_training_visualizations"
        self._count = 0

        self.compute_format_rewards = format_rewards_v2
        self.compute_answer_rewards = answer_rewards_v3
        self.matcher = MaskMatcher()



    def _add_special_tokens(self, tokenizer, special_tokens):
        self.mllm.add_special_tokens(tokenizer, special_tokens)
        self.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0] # required to make add_special_tokens to be False to avoid <bos> or <eos>

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        return super().load_state_dict(state_dict, strict, assign)

    def _merge_lora(self):
        if isinstance(self.mllm.model, PeftModelForCausalLM):
            self.mllm.model = self.mllm.model.merge_and_unload()
            return
        
        try:
            self.mllm.model.language_model = self.mllm.model.language_model.merge_and_unload()
        except:
            print("Skip language model, no LoRA in it !!!")
        try:
            self.mllm.model.vision_model = self.mllm.model.vision_model.merge_and_unload()
        except:
            print("Skip vision encoder, no LoRA in it !!!")
        return

    def all_state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict

    def state_dict(self, *args, **kwargs):
        prefix = kwargs.pop('prefix', '')
        state_dict_mllm = self.mllm.state_dict(*args, prefix=prefix + 'mllm.', **kwargs)
        state_dict_sam2 = self.grounding_encoder.state_dict(*args, prefix=prefix + 'grounding_encoder.', **kwargs)
        state_dict_text = self.text_hidden_fcs.state_dict(*args, prefix=prefix + 'text_hidden_fcs.', **kwargs)
        to_return = OrderedDict()
        to_return.update(state_dict_mllm)
        to_return.update(
            {k: v
             for k, v in state_dict_sam2.items() if k.startswith('grounding_encoder.sam2_model.sam_mask_decoder')})
        to_return.update(state_dict_text)
        return to_return

    def check_obj_number(self, pred_embeddings_list_video, gt_masks_video, fix_number=5):
        assert len(pred_embeddings_list_video) == len(gt_masks_video)
        ret_pred_embeddings_list_video = []
        ret_gt_masks_video = []
        for pred_mebeds, gt_masks in zip(pred_embeddings_list_video, gt_masks_video):
            # assert len(pred_mebeds) == len(gt_masks)
            if len(pred_mebeds) != len(gt_masks):
                min_num = min(len(pred_mebeds), len(gt_masks))
                pred_mebeds = pred_mebeds[:min_num]
                gt_masks = gt_masks[:min_num]
            if len(pred_mebeds) != fix_number:
                if len(pred_mebeds) > fix_number:
                    _idxs = torch.randperm(pred_mebeds.shape[0])
                    _idxs = _idxs[:fix_number]
                    pred_mebeds = pred_mebeds[_idxs]
                    gt_masks = gt_masks[_idxs]
                else:
                    n_repeat = fix_number // len(pred_mebeds) + 1
                    pred_mebeds = torch.cat([pred_mebeds] * n_repeat, dim=0)[:fix_number]
                    gt_masks = torch.cat([gt_masks] * n_repeat, dim=0)[:fix_number]
            ret_pred_embeddings_list_video.append(pred_mebeds)
            ret_gt_masks_video.append(gt_masks)
        return ret_pred_embeddings_list_video, ret_gt_masks_video

    def _get_pesudo_data(self, dtype, device):
        g_pixel_values = torch.zeros((3, 1024, 1024), dtype=dtype, device=device)
        g_pixel_values = [g_pixel_values] * self.bs
        frames_per_batch = [1] * self.bs
        gt_masks = torch.zeros((5, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return g_pixel_values, frames_per_batch, gt_masks
    

    def group_generation_optimized(self, input_ids, pixel_values, g_pixel_values, gt_masks, image_grid_thw=None, data_samples=None):
        """Optimized group generation that works with preprocessed data"""
        # Use the first batch for generation
        input_ids_for_gen = input_ids[0].unsqueeze(0)

        # pixel_values is a list, use the first, the first along bs; num_image, three, h, w
        pixel_values = pixel_values[0] 
        
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw[0]


        gt_masks = gt_masks[0]

        # Create attention mask
        attention_mask = torch.ones_like(input_ids_for_gen, dtype=torch.bool)
        
        mm_inputs = {
            'pixel_values': pixel_values,
            'input_ids': input_ids_for_gen,
            'attention_mask': attention_mask,
            'image_grid_thw': image_grid_thw,
            'position_ids': None,
            'past_key_values': None,
            'labels': None
        }

        # Generate multiple sequences
        self.eval()
        with torch.no_grad():
            generate_output = self.mllm.generate(
                **mm_inputs,
                generation_config=self.gen_config,
                streamer=None,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        # self.train()

        # Process generation results
        prompt_length = input_ids_for_gen.shape[-1]
        output_results = []

        with torch.no_grad():
            g_pixels = torch.stack([
                self.grounding_encoder.preprocess_image(pixel.to(self.torch_dtype)) for pixel in g_pixel_values
            ]).to(generate_output.sequences.device)
            sam_feats = self.grounding_encoder.get_sam2_feats(g_pixels)
        
        for i_gen in range(self.generation_nums):
            output_ids = generate_output.sequences[i_gen]
            # Get hidden states for segmentation
            last_hidden_states = [item[-1][i_gen] for item in generate_output.hidden_states]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            
            # Extract SEG token embeddings
            seg_hidden_states = get_seg_hidden_states(
                last_hidden_states, 
                output_ids[:-1], 
                seg_id=self.seg_token_idx
            )

            output_ids = output_ids[prompt_length:]

            output_mask = output_ids != self.tokenizer.pad_token_id
            output_ids = output_ids[output_mask]
            # Decode the generated text
            output_text = self.tokenizer.decode(output_ids, skip_special_tokens=False)
            
            # Process masks if SEG tokens exist
            with torch.no_grad():
                if len(seg_hidden_states) > 0:
                    seg_embeddings = self.text_hidden_fcs(seg_hidden_states)
                    num_objs = len(seg_embeddings)
                    language_embeddings = seg_embeddings[:, None]
                    
                    sam_states = self.grounding_encoder.get_sam2_states_from_feats(copy.deepcopy(sam_feats), expand_size=num_objs)
                    pred_masks = self.grounding_encoder.inject_language_embd(
                        sam_states, language_embeddings, nf_nobj=(1, num_objs)
                    )
                    pred_masks = pred_masks[0]
                    cur_pred_mask = pred_masks.detach()
                else:
                    cur_pred_mask = None
            
            output_results.append({
                'prompt_ids': input_ids_for_gen,
                'answer_ids': output_ids.unsqueeze(0),
                'predict_text': output_text,
                'predict_masks': cur_pred_mask,
            })

        # Compute rewards
        predictions_texts = [result['predict_text'] for result in output_results]
        predictions_masks = [result['predict_masks'] for result in output_results]

        # Calculate answer [SEG]s
        answer_idxs = []
        for pred_text in predictions_texts:
            _, answer_seg_idx = find_seg_indices(pred_text)
            answer_idxs.append(answer_seg_idx)

        format_rewards = self.compute_format_rewards(predictions_texts)

        answer_rewards = self.compute_answer_rewards(predictions_masks, gt_masks, answer_idxs=answer_idxs)

        format_rewards = torch.tensor(format_rewards, dtype=torch.float32)
        answer_rewards = torch.tensor(answer_rewards, dtype=torch.float32)

        rewards = format_rewards * self.format_reward_weight + answer_rewards * self.answer_reward_weight
        rewards = rewards.cuda()
        mean_grouped_rewards = rewards.view(-1, self.generation_nums).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.generation_nums).std(dim=1)

        # Normalize rewards to compute advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.generation_nums, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.generation_nums, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Prepare input for forward pass
        prompt_ids = [result['prompt_ids'] for result in output_results]
        answer_ids = [result['answer_ids'] for result in output_results]
        
        prompt_lengths = [item.shape[-1] for item in prompt_ids]
        answer_lengths = [item.shape[-1] for item in answer_ids]
        overall_length = [pl + al for pl, al in zip(prompt_lengths, answer_lengths)]
        max_length = max(overall_length)

        input_ids = torch.zeros((self.generation_nums, max_length), dtype=torch.int64).to(prompt_ids[0].device) + DEFAULT_PAD_TOKEN_INDEX
        attention_mask = torch.zeros_like(input_ids).bool()

        for i in range(self.generation_nums):
            cur_input_ids = torch.cat([prompt_ids[i], answer_ids[i]], dim=-1)
            attention_mask[i, :cur_input_ids.shape[-1]] = True
            input_ids[i:i+1, :cur_input_ids.shape[-1]] = cur_input_ids
        
        position_ids = torch.arange(max_length).unsqueeze(0).long().repeat(self.generation_nums, 1)

        reward_log_infos = {
            'max_format_reward': torch.max(format_rewards).detach(), 
            'max_answer_reward': torch.max(answer_rewards).detach(), 
            'max_reward': torch.max(rewards).detach(),
        }

        # print for the main process
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"Reward list: {rewards.cpu().numpy().tolist()}")
            print(f"Format reward list: {format_rewards.cpu().numpy().tolist()}")
            print(f"Answer reward list: {answer_rewards.cpu().numpy().tolist()}")


        # Visualization: Save results every 10 steps
        self._count += 1
        if self.visualization_step_interval > 0:
            if self._count % self.visualization_step_interval == 0:
                self._save_group_visualizations(
                    pixel_values, predictions_texts, predictions_masks, 
                    rewards, format_rewards, answer_rewards, data_samples
                )
        self.train()
        return {
            'advantages': advantages,
            'prompt_length': prompt_length,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'reward_log_infos': reward_log_infos,
            'pred_texts': predictions_texts,
        }

    def _save_group_visualizations(self, pixel_values, predictions_texts, predictions_masks, 
                                 rewards, format_rewards, answer_rewards, data_samples):
        """Save JSON records for each group member every 10 steps"""
        # Create output directory
        os.makedirs(self.visualization_dir, exist_ok=True)
        step_dir = os.path.join(self.visualization_dir, f"step_{self._count}")
        os.makedirs(step_dir, exist_ok=True)
        
        # Get idx from data_samples['metainfo'] - it's a list of dicts
        idx = None
        if data_samples is not None and 'metainfo' in data_samples:
            metainfo_list = data_samples['metainfo']
            if metainfo_list and len(metainfo_list) > 0:
                # Use the first item's key (assuming batch size 1 for generation)
                idx = metainfo_list[0].get('key', None)
        
        # Create records for each group member
        records = []
        for i in range(len(predictions_texts)):
            pred_text = predictions_texts[i]
            pred_masks = predictions_masks[i]
            reward = rewards[i].item() if hasattr(rewards[i], 'item') else rewards[i]
            format_reward = format_rewards[i].item() if hasattr(format_rewards[i], 'item') else format_rewards[i]
            answer_reward = answer_rewards[i].item() if hasattr(answer_rewards[i], 'item') else answer_rewards[i]
            
            # Convert masks to serializable format if they exist
            masks_data = None
            if pred_masks is not None:
                if isinstance(pred_masks, torch.Tensor):
                    masks_np = pred_masks.cpu().numpy()
                else:
                    masks_np = pred_masks
                
                # Convert to RLE format for efficient storage
                if len(masks_np.shape) == 3:  # [num_masks, H, W]
                    masks_data = mask_to_rle(masks_np)
                else:  # Single mask [H, W]
                    masks_data = mask_to_rle([masks_np])
            
            # Create record for this group member
            record = {
                'group_member_id': i,
                'idx': idx,
                'masks': masks_data,
                'rewards': reward,
                'format_rewards': format_reward,
                'answer_rewards': answer_reward,
                'answers': pred_text,
                'step': self._count
            }
            records.append(record)
        
        # Save all records to a single JSON file
        json_filename = f"step_{self._count}_group_results.json"
        json_path = os.path.join(step_dir, json_filename)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"Saved {len(records)} group records to: {json_path}")
        except Exception as e:
            print(f"Error saving JSON records: {e}")

    def _get_per_token_logps(self, logits, input_ids):
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def forward(self, data, data_samples=None, mode='loss'):
        # Adapt to sa2va.py input format
        g_pixel_values = data.pop('g_pixel_values', None)
        pixel_values = data.pop('pixel_values', None)
        image_grid_thw = data.pop('image_grid_thw', None)
        gt_masks = data.pop('masks', None)
        input_ids = data['input_ids']

        if gt_masks is None:
            raise ValueError("gt_masks is required for training.")

        # Use the same data structure as sa2va.py for training
        with torch.no_grad():
            result = self.group_generation_optimized(input_ids, pixel_values, g_pixel_values, gt_masks, image_grid_thw, data_samples)
            advantages = result['advantages']
            prompt_length = result['prompt_length']
            input_ids = result['input_ids']
            attention_mask = result['attention_mask']
            position_ids = result['position_ids']
            reward_log_infos = result['reward_log_infos']

            group_text = result['pred_texts']
        
        position_ids = position_ids.to(device='cuda')
        seg_valid = True  # Set seg_valid flag

        output = self.mllm({
            # .unsqueeze(0) on pixel_values to match batch size
            'pixel_values': [pixel_values[0]] * self.generation_nums, 
            'image_grid_thw': [image_grid_thw[0]] * self.generation_nums if image_grid_thw is not None else None,
            'input_ids': input_ids, 'position_ids': position_ids,
            'attention_mask': attention_mask, 'labels': None,
        })

        per_token_logps = self._get_per_token_logps(logits=output.logits, input_ids=input_ids)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        ratio = torch.exp(per_token_logps - per_token_logps.detach())
        per_token_loss = ratio * advantages.unsqueeze(1)
        per_token_loss = -per_token_loss

        completion_mask = attention_mask[:, prompt_length:]
        completion_mask = completion_mask.to(per_token_loss)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # print loss
        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(f"Policy loss: {loss.item():.4f}")


        if self.train_segmentation and seg_valid:
            # Process gt_masks - handle list format
            if isinstance(gt_masks, list):
                gt_mask = gt_masks[0]  # Use first mask in batch
            else:
                gt_mask = gt_masks
                
            mask_shape = gt_mask.shape[-2:]
            best_batch_idx = torch.argmax(advantages).item()

            best_input_ids = input_ids[best_batch_idx:best_batch_idx+1]
            seg_token_mask = best_input_ids == self.seg_token_idx

            num_objs = seg_token_mask.sum().item()
            none_seg = False
            if num_objs == 0:
                seg_token_mask[0][0] = True
                num_objs = 1
                none_seg = True

            hidden_states = output.hidden_states
            seg_embeddings = hidden_states[-1][best_batch_idx:best_batch_idx+1][seg_token_mask]
            pred_embeddings = self.text_hidden_fcs(seg_embeddings)
            language_embeddings = pred_embeddings[:, None]

            # Process g_pixel_values - handle list format
            if isinstance(g_pixel_values, list):
                g_pixel_val = g_pixel_values[0]
            else:
                g_pixel_val = g_pixel_values
                
            g_pixel_processed = torch.stack([
                self.grounding_encoder.preprocess_image(g_pixel_val)
            ])
            num_frames = 1
            
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_processed, expand_size=num_objs)
            pred_masks = self.grounding_encoder.inject_language_embd(sam_states, language_embeddings, nf_nobj=(num_frames, num_objs))

            bs = len(pred_masks)
            assert bs == num_frames
            loss_mask, loss_dice = 0, 0
            accuracy = 0

            pred_text = self.tokenizer.decode(best_input_ids[0][prompt_length:], skip_special_tokens=False)
            # calculate the answer [SEG]s
            reason_idx, _ = find_seg_indices(pred_text)
            for reason_seg_id in reason_idx:
                if reason_seg_id < pred_masks.shape[1]:  # Safety check
                    pred_masks[0, reason_seg_id] *= .0

            # Merge - handle mask dimensions
            if pred_masks.dim() > 3:  # (bs, num_seg, h, w)
                pred_masks = pred_masks.max(dim=1).values
            
            # Ensure gt_mask has correct format
            if gt_mask.dim() > 2:  # (num_seg, h, w)
                gt_mask = gt_mask.max(dim=0, keepdim=True).values
            else:  # (h, w) -> (1, h, w)
                gt_mask = gt_mask.unsqueeze(0)

            pred_masks = F.interpolate(pred_masks.unsqueeze(0), size=mask_shape, mode='bilinear').squeeze(0)

            if self.loss_sample_points:
                sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_masks, gt_mask)
                sam_loss_dice = self.loss_dice(
                    sampled_pred_mask,
                    sampled_gt_mask, avg_factor=(len(gt_mask) + 1e-4))
                sam_loss_mask = self.loss_mask(
                    sampled_pred_mask.reshape(-1),
                    sampled_gt_mask.reshape(-1),
                    avg_factor=(pred_masks.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
            else:
                sam_loss_mask = self.loss_mask(pred_masks, gt_mask)
                sam_loss_dice = self.loss_dice(pred_masks, gt_mask)
            accuracy += torch.eq((pred_masks.sigmoid() > 0.5), gt_mask).to(pred_masks).mean()
            loss_mask += sam_loss_mask
            loss_dice += sam_loss_dice

            if none_seg:
                seg_ratio = 0.0
            else:
                seg_ratio = 1.0
            loss_dict = {
                'loss_mask': loss_mask / (bs + 1e-4) * seg_ratio * self.seg_ratio,
                'loss_dice': loss_dice / (bs + 1e-4) * seg_ratio * self.seg_ratio,
                'loss_grpo': loss,
            }
            loss_dict.update(reward_log_infos)
        else:
            loss_dict = {
                'loss_grpo': loss,
            }
            loss_dict.update(reward_log_infos)

        return loss_dict


    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def generate_video_pred_embeddings(self, pred_embeddings_list, frames_per_batch):
        assert len(pred_embeddings_list) == len(frames_per_batch)
        pred_embeddings_list_video = []
        for pred_embedding_batch, frame_nums in zip(pred_embeddings_list, frames_per_batch):
            pred_embeddings_list_video += [pred_embedding_batch] * frame_nums
        return pred_embeddings_list_video

    def process_video_gt_masks(self, gt_masks, frames_per_batch):
        gt_masks_video = []

        assert len(gt_masks) == len(frames_per_batch)
        for gt_masks_batch, frames_num in zip(gt_masks, frames_per_batch):
            N, H, W = gt_masks_batch.shape
            assert N % frames_num == 0
            gt_masks_batch = gt_masks_batch.reshape(
                N // frames_num, frames_num, H, W)
            for i in range(frames_num):
                gt_masks_video.append(gt_masks_batch[:, i])
        return gt_masks_video

    def preparing_for_generation(self, metainfo, **kwargs):
        raise NotImplementedError("Sa2VA does not support preparing for generation, please use predict_video instead.")

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
