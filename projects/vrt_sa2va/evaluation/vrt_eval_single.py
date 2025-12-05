"""
VER (Visual Evidence Reasoning) Evaluation Script
Evaluates LLM performance on VER benchmark by comparing predicted masks with human-labeled ground truth.
"""

import argparse
import glob
import json
import os
import torch
import torch.multiprocessing as mp
import tqdm
from pycocotools import mask as _mask
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Dict, List

from projects.vrt_sa2va.utils.strings import find_seg_indices
from projects.vrt_sa2va.evaluation.vrt_eval_dataloader import VRTEvalDataset


def mask_to_rle(mask):
    """Convert mask to RLE format"""
    rle = []
    for m in mask:
        rle.append(_mask.encode(np.asfortranarray(m.astype(np.uint8))))
        rle[-1]['counts'] = rle[-1]['counts'].decode()
    return rle


def calculate_miou(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> List[float]:
    """
    Calculate IoU for each ground truth mask against the best matching predicted mask
    
    Args:
        pred_masks: List of predicted masks (H, W)
        gt_masks: List of ground truth masks (H, W)
    """

    if not gt_masks:
        return []

    if not pred_masks:
        return [0.0] * len(gt_masks)
    
    # Stack masks into 3D arrays for vectorized operations
    pred_masks_np = np.stack(pred_masks)
    gt_masks_np = np.stack(gt_masks)

    # Expand dims for broadcasting
    pred_masks_np = pred_masks_np[:, np.newaxis, :, :]
    gt_masks_np = gt_masks_np[np.newaxis, :, :, :]

    # Calculate intersection and union matrices
    intersection = np.logical_and(pred_masks_np, gt_masks_np).sum(axis=(2, 3))
    union = np.logical_or(pred_masks_np, gt_masks_np).sum(axis=(2, 3))

    # Calculate IoU matrix, handling division by zero
    iou_matrix = np.zeros_like(intersection, dtype=np.float32)
    np.divide(intersection, union, out=iou_matrix, where=union > 0)

    # Use linear assignment to find the best one-to-one matching
    # The cost matrix is 1 - IoU to maximize the IoU sum
    row_ind, col_ind = linear_sum_assignment(1 - iou_matrix)
    
    # Create a list of IoUs for each ground truth mask
    # Initialize with zeros, so un-matched GTs have an IoU of 0
    assigned_ious = np.zeros(gt_masks_np.shape[1])
    assigned_ious[col_ind] = iou_matrix[row_ind, col_ind]

    assert len(assigned_ious) == len(gt_masks), "Mismatch in number of GT masks"

    return assigned_ious.tolist()


def prepare_question_prompt(question: str, use_thinking: bool = True) -> str:
    """
    Prepare the question prompt for the LLM following the VER format
    Uses the same template as R2SDataset for consistency
    
    Args:
        question: The question to ask
        use_thinking: Whether to use thinking process or simple prompt
        
    Returns:
        Formatted prompt string
    """
    if use_thinking:
        # Use the same template as R2SDataset with thinking
        THINK_TEMPLATE = '<image>' + "\n" + "{sent}\n\nYou should first think about the reasoning process in the mind and then provides the user with the answer. Please respond with segmentation mask in both the thinking process and the answer."
        TEMPLATE_TEMPLATE = 'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answers here </answer>.'
        
        prompt = THINK_TEMPLATE.format(sent=question)
        prompt = "{question}\n\n{template}".format(question=prompt, template=TEMPLATE_TEMPLATE)
    else:
        # Simple prompt without thinking
        prompt = '<image>' + "\n" + "{sent}\n\nPlease respond with segmentation mask.".format(sent=question)
    
    return prompt


class VERLLMEvaluator:
    """Evaluator for VER benchmark using LLM"""
    
    def __init__(self, model_path: str, dataset: VRTEvalDataset = None, use_thinking: bool = True, device: int = 0):
        """
        Initialize the evaluator
        
        Args:
            model_path: Path to the LLM model
            dataset: VER dataset instance
            use_thinking: Whether to use thinking process in prompts
        """
        self.model_path = model_path
        self.dataset = dataset
        self.use_thinking = use_thinking
        self.device = device

        print(f"Loading model from {model_path} on cuda:{device}...")
        # prepare device
        self.torch_device = torch.cuda.current_device()
        

        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        # Load model onto the assigned device
        self.model = AutoModel.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map={'': self.torch_device},
            use_flash_attn=True,
            trust_remote_code=True,
        ).eval()


        if 'qwen' in model_path.lower() or 'q3' in model_path.lower():
            print("Using Qwen model processor...")
            self.tokenizer = None
            self.preprocessor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        else:
            print("Using non-Qwen model tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self.preprocessor = None
        if hasattr(self.model, 'preparing_for_generation'):
            self.model.preparing_for_generation(self.tokenizer, max_new_tokens=512)
        print("Model loaded successfully!")
    
    def evaluate_sample(self, sample: Dict) -> Dict:
        """
        Evaluate a single sample
        
        Args:
            sample: Sample data containing image, question, and ground truth
            
        Returns:
            Evaluation results for this sample
        """
        question = sample['question']
        image = sample['image']
        key = sample['key']
        
        # Prepare the prompt
        prompt = prepare_question_prompt(question, self.use_thinking)
        
        # Get model prediction
        # try:
        data_batch = {
            'image': image,
            'text': prompt
        }
        
        pred = self.model.predict_forward(
            **data_batch, 
            tokenizer=self.tokenizer, processor=self.preprocessor
        )
        pred_masks = pred['prediction_masks']  # List of masks
        pred_text = pred['prediction']
        
        print(f"Sample {key}: {pred_text[:100]}...")

        reasoning_masks = []
        answer_masks = []
        
        if len(pred_masks) > 0:
            if self.use_thinking:
                # Use seg indices for thinking mode
                reasoning_seg_idx, answer_seg_idx = find_seg_indices(pred_text)
                
                if len(reasoning_seg_idx) > 0:
                    reasoning_masks = [pred_masks[i] for i in reasoning_seg_idx if i < len(pred_masks)]
                
                if len(answer_seg_idx) > 0:
                    answer_masks = [pred_masks[i] for i in answer_seg_idx if i < len(pred_masks)]
            else:
                # For non-thinking mode, treat all masks as answer masks
                answer_masks = pred_masks
        
        # Flatten masks (remove batch dimension if present)
        if reasoning_masks:
            reasoning_masks = [mask.squeeze() if mask.ndim > 2 else mask for mask in reasoning_masks]
        if answer_masks:
            answer_masks = [mask.squeeze() if mask.ndim > 2 else mask for mask in answer_masks]
        
        return {
            'key': key,
            'question': question,
            'prediction_text': pred_text,
            'reasoning_masks': reasoning_masks,
            'answer_masks': answer_masks,
            'num_pred_masks': len(pred_masks),
            'num_reasoning_masks': len(reasoning_masks),
            'num_answer_masks': len(answer_masks),
        }
    
    def evaluate_all_samples(self, output_dir: str, 
                             samples: List[Dict] = None, 
                             rank: int = None) -> Dict:
        """
        Evaluate all samples in the dataset
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Complete evaluation results
        """
        # Get evaluation samples (either provided or from dataset)
        if samples is None:
            if self.dataset is None:
                raise ValueError("No samples provided and evaluator has no dataset.")
            evaluation_samples = self.dataset.get_evaluation_samples()
        else:
            evaluation_samples = samples

        print(f"Found {len(evaluation_samples)} evaluation samples (rank={rank})")
        
        os.makedirs(output_dir, exist_ok=True)
        

        # collect the following information among the samples
        results = []
        reasoning_ious = []
        answer_ious = []

        per_cls_reasoning_ious = {}
        per_cls_answer_ious = {}
        
        # Track counts per category
        per_cls_sample_count = {}
        per_cls_reasoning_mask_count = {}
        per_cls_answer_mask_count = {}
        
        for i, sample in enumerate(tqdm.tqdm(evaluation_samples, desc="Evaluating")):
            pred_result = self.evaluate_sample(sample)
            
            gt_reasoning_obj_ids = sample.get('human_labeled_r_objs', [])
            gt_answer_obj_ids = sample.get('human_labeled_a_objs', [])
            
            gt_reasoning_masks = []
            gt_answer_masks = []
            
            objects_info = sample.get('objects_info', {})

            # Extract ground truth masks
            for obj_id in gt_reasoning_obj_ids:
                if obj_id in objects_info:
                    gt_reasoning_masks.append(objects_info[obj_id]['mask'])
            assert len(gt_reasoning_masks) == len(gt_reasoning_obj_ids), "Mismatch in reasoning GT masks extraction"

            for obj_id in gt_answer_obj_ids:
                if obj_id in objects_info:
                    gt_answer_masks.append(objects_info[obj_id]['mask'])
            assert len(gt_answer_masks) == len(gt_answer_obj_ids), "Mismatch in answer GT masks extraction"

            # Calculate IoU scores - now returns list of IoUs for each GT mask
            sample_reasoning_ious = []
            sample_answer_ious = []

            sample_reasoning_ious = calculate_miou(pred_result.get('reasoning_masks', []), gt_reasoning_masks)
            assert len(sample_reasoning_ious) == len(gt_reasoning_masks), "Mismatch in reasoning IoU calculation"
            reasoning_ious.extend(sample_reasoning_ious)


            for cls_id in sample.get('class_ids', []):
                if cls_id not in per_cls_reasoning_ious:
                    per_cls_reasoning_ious[cls_id] = []
                per_cls_reasoning_ious[cls_id].extend(sample_reasoning_ious)
                
                # Count samples and masks per category
                if cls_id not in per_cls_sample_count:
                    per_cls_sample_count[cls_id] = 0
                per_cls_sample_count[cls_id] += 1
                
                if cls_id not in per_cls_reasoning_mask_count:
                    per_cls_reasoning_mask_count[cls_id] = 0
                per_cls_reasoning_mask_count[cls_id] += len(sample_reasoning_ious)


            sample_answer_ious = calculate_miou(pred_result.get('answer_masks', []), gt_answer_masks)
            assert len(sample_answer_ious) == len(gt_answer_masks), "Mismatch in answer IoU calculation"
            answer_ious.extend(sample_answer_ious)

            for cls_id in sample.get('class_ids', []):
                if cls_id not in per_cls_answer_ious:
                    per_cls_answer_ious[cls_id] = []
                per_cls_answer_ious[cls_id].extend(sample_answer_ious)
                
                if cls_id not in per_cls_answer_mask_count:
                    per_cls_answer_mask_count[cls_id] = 0
                per_cls_answer_mask_count[cls_id] += len(sample_answer_ious)

            # Calculate mean IoU for this sample for reporting
            reasoning_miou = np.mean(sample_reasoning_ious) if sample_reasoning_ious else 0.0
            answer_miou = np.mean(sample_answer_ious) if sample_answer_ious else 0.0

            result_entry = {
                'sample_info': {
                    'key': sample['key'],
                    'question': sample['question'],
                    'human_confidence': sample.get('human_confidence', 0),
                    'class_ids': sample.get('class_ids', []),
                },
                'ground_truth': {
                    'reasoning_obj_ids': gt_reasoning_obj_ids,
                    'answer_obj_ids': gt_answer_obj_ids,
                    'num_reasoning_masks': len(gt_reasoning_masks),
                    'num_answer_masks': len(gt_answer_masks),
                },
                'prediction': {
                    'text': pred_result['prediction_text'],
                    'num_pred_masks': pred_result['num_pred_masks'],
                    'num_reasoning_masks': pred_result['num_reasoning_masks'],
                    'num_answer_masks': pred_result['num_answer_masks'],
                    'reasoning_masks_rle': mask_to_rle(pred_result['reasoning_masks']) if pred_result['reasoning_masks'] else [],
                    'answer_masks_rle': mask_to_rle(pred_result['answer_masks']) if pred_result['answer_masks'] else [],
                },
                'evaluation': {
                    'reasoning_miou': reasoning_miou,
                    'answer_miou': answer_miou,
                },
                'evaluation_details': {
                    'reasoning_ious': sample_reasoning_ious,
                    'answer_ious': sample_answer_ious,
                },
                'error': pred_result.get('error', '')
            }
            
            results.append(result_entry)
            
            if (i + 1) % 10 == 0:
                current_r_miou = np.mean(reasoning_ious) if reasoning_ious else 0.0
                current_a_miou = np.mean(answer_ious) if answer_ious else 0.0
                print(f"Progress {i+1}/{len(evaluation_samples)}: "
                      f"Reasoning mIoU: {current_r_miou:.3f}, "
                      f"Answer mIoU: {current_a_miou:.3f}")
        

        reasoning_ious = np.array(reasoning_ious)
        answer_ious = np.array(answer_ious)
        
        # Calculate final metrics
        final_reasoning_miou = np.mean(reasoning_ious) if len(reasoning_ious) > 0 else 0.0
        final_answer_miou = np.mean(answer_ious) if len(answer_ious) > 0 else 0.0

        final_r_lq = (reasoning_ious > 0.5).sum() / len(reasoning_ious) if len(reasoning_ious) >0 else 0.0
        final_r_sq = reasoning_ious[reasoning_ious > 0.5].mean() if (reasoning_ious > 0.5).sum() >0 else 0.0

        final_a_lq = (answer_ious > 0.5).sum() / len(answer_ious) if len(answer_ious) >0 else 0.0
        final_a_sq = answer_ious[answer_ious > 0.5].mean() if (answer_ious > 0.5).sum() >0 else 0.0


        # Callculate per-class metrics
        per_cls_metrics = {}
        for cls_id, ious in per_cls_reasoning_ious.items():
            ious_np = np.array(ious)
            cls_r_miou = np.mean(ious_np) if len(ious_np) > 0 else 0.0
            cls_r_lq = (ious_np > 0.5).sum() / len(ious_np) if len(ious_np) >0 else 0.0
            cls_r_sq = ious_np[ious_np > 0.5].mean() if (ious_np > 0.5).sum() >0 else 0.0


            per_cls_metrics[cls_id] = {
                'reasoning_miou': cls_r_miou,
                'reasoning_lq': cls_r_lq,
                'reasoning_sq': cls_r_sq,
                'sample_count': per_cls_sample_count.get(cls_id, 0),
                'reasoning_mask_count': per_cls_reasoning_mask_count.get(cls_id, 0),
            }

        for cls_id, ious in per_cls_answer_ious.items():
            ious_np = np.array(ious)
            cls_a_miou = np.mean(ious_np) if len(ious_np) > 0 else 0.0
            cls_a_lq = (ious_np > 0.5).sum() / len(ious_np) if len(ious_np) >0 else 0.0
            cls_a_sq = ious_np[ious_np > 0.5].mean() if (ious_np > 0.5).sum() >0 else 0.0

            per_cls_metrics[cls_id].update({
                'answer_miou': cls_a_miou,
                'answer_lq': cls_a_lq,
                'answer_sq': cls_a_sq,
                'answer_mask_count': per_cls_answer_mask_count.get(cls_id, 0),
            })
        
        evaluation_summary = {
            'model_path': self.model_path,
            'use_thinking': self.use_thinking,
            'total_samples': len(evaluation_samples),
            'total_reasoning_masks_evaluated': len(reasoning_ious),
            'total_answer_masks_evaluated': len(answer_ious),
            'metrics': {
                'reasoning_miou': final_reasoning_miou,
                'answer_miou': final_answer_miou,
                
                # new metrics
                'reasoning_lq': final_r_lq,
                'reasoning_sq': final_r_sq,
                'answer_lq': final_a_lq,
                'answer_sq': final_a_sq, 
            },
            'detailed_results': results,
            'per_cls_metrics': per_cls_metrics,
        }
        
        # Save results (per-rank if rank provided)
        os.makedirs(output_dir, exist_ok=True)
        if rank is None:
            results_file = os.path.join(output_dir, 'ver_evaluation_results.json')
        else:
            results_file = os.path.join(output_dir, f'ver_evaluation_results_rank{rank}.json')

        # Include raw ious and per-class lists so master process can merge them
        evaluation_summary['raw_reasoning_ious'] = reasoning_ious.tolist() if isinstance(reasoning_ious, np.ndarray) else reasoning_ious
        evaluation_summary['raw_answer_ious'] = answer_ious.tolist() if isinstance(answer_ious, np.ndarray) else answer_ious
        evaluation_summary['per_cls_reasoning_ious'] = per_cls_reasoning_ious
        evaluation_summary['per_cls_answer_ious'] = per_cls_answer_ious
        evaluation_summary['per_cls_sample_count'] = per_cls_sample_count
        evaluation_summary['per_cls_reasoning_mask_count'] = per_cls_reasoning_mask_count
        evaluation_summary['per_cls_answer_mask_count'] = per_cls_answer_mask_count
        with open(results_file, 'w') as f:
            json.dump(evaluation_summary, f, indent=2)

        return evaluation_summary


def derive_output_dir_from_model_path(model_path: str) -> str:
    """
    Derive output directory from model path by removing '_hf' suffix and using work_dirs prefix
    
    Args:
        model_path: Path to the model
        
    Returns:
        Output directory path
    """
    import os
    model_name = os.path.basename(model_path.rstrip('/'))
    # Remove '_hf' suffix if present
    if model_name.endswith('_hf'):
        model_name = model_name[:-3]
    
    model_name = model_name + '_vrt_eval_results'
    return os.path.join('work_dirs', model_name)


def print_evaluation_summary(evaluation_summary: Dict):
    """Print a concise evaluation summary (moved out of evaluator for single/multi-process use)."""
    print(f"\n=== VER Evaluation Results ===")
    print(f"Model: {evaluation_summary.get('model_path', '')}")
    print(f"Total samples: {evaluation_summary.get('total_samples', 0)}")
    metrics = evaluation_summary.get('metrics', {})
    reasoning_miou = metrics.get('reasoning_miou', 0.0)
    answer_miou = metrics.get('answer_miou', 0.0)
    total_r = evaluation_summary.get('total_reasoning_masks_evaluated', 0)
    total_a = evaluation_summary.get('total_answer_masks_evaluated', 0)
    print(f"Reasoning mIoU: {reasoning_miou:.3f} (on {total_r} GT masks)")
    print(f"Answer mIoU: {answer_miou:.3f} (on {total_a} GT masks)")

    print(f"Metrics (reasoning_lq, reasoning_sq, answer_lq, answer_sq):")
    print(f"{(metrics.get('reasoning_lq',0.0) * 100):.1f}, {(metrics.get('reasoning_sq',0.0) * 100):.1f}, {(metrics.get('answer_lq',0.0) * 100):.1f}, {(metrics.get('answer_sq',0.0) * 100):.1f}")

    print("\nPer-class metrics:")
    per_cls_metrics = evaluation_summary.get('per_cls_metrics', {})
    for cls_id, metrics in per_cls_metrics.items():
        sample_count = metrics.get('sample_count', 0)
        r_mask_count = metrics.get('reasoning_mask_count', 0)
        a_mask_count = metrics.get('answer_mask_count', 0)
        print(f"Class {cls_id} (samples: {sample_count}, reasoning_masks: {r_mask_count}, answer_masks: {a_mask_count}):")
        print(f"  Metrics (reasoning_lq, reasoning_sq, answer_lq, answer_sq, reasoning_miou, answer_miou):")
        print(f"  {(metrics.get('reasoning_lq',0.0) * 100):.1f}, {(metrics.get('reasoning_sq',0.0) * 100):.1f}, "
              f"{(metrics.get('answer_lq',0.0) * 100):.1f}, {(metrics.get('answer_sq',0.0) * 100):.1f}, "
              f"{(metrics.get('reasoning_miou',0.0) * 100):.1f}, {(metrics.get('answer_miou',0.0) * 100):.1f}")


def parse_args():
    parser = argparse.ArgumentParser(description='VER Evaluation Script')
    parser.add_argument(
        'model_path',
        help='Path to the LLM model')
    parser.add_argument(
        '--tfrecord_path',
        default='data/VER_BENCH/VER-Labeling/data/0801_data_label/sa2va_samples_*.tfrecord',
        help='Path to the directory containing TFRecord files'
    )
    parser.add_argument(
        '--human_label_path',
        default='data/VER_BENCH/VER-Labeling/data/0801_data_label_result/sa2va_samples_*.txt',
        help='Path to the directory containing human-labeled results files'
    )
    parser.add_argument(
        '--output_dir',
        default=None,
        help='Directory to save evaluation results (default: derived from model path)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to evaluate (for testing)'
    )
    parser.add_argument(
        '--no-thinking',
        action='store_true',
        help='Use simple prompt without thinking process'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='Number of GPUs to use for evaluation (spawn one process per GPU)'
    )
    
    return parser.parse_args()


# worker entry that each process will run
def worker_entry(rank, args, output_dir, use_thinking):
    # set device
    torch.cuda.set_device(rank)

    # do not need the distributed backend as we do not need to communicate
    # dist.init_process_group("nccl", rank=rank, world_size=world_size)

    print(f"Worker {rank} starting on cuda:{rank}.... Loading the dataset first.")

    # Initialize dataset within the worker
    tfrecord_paths = glob.glob(args.tfrecord_path)
    human_label_paths = glob.glob(args.human_label_path)
    dataset = VRTEvalDataset(
        tfrecord_paths, human_label_paths, 
        cls_mapping='data/VER_BENCH/VER-Labeling/data/ver_labeling_1106.json'
    )
    # each worker will take a strided subset of samples
    evaluation_samples = dataset.get_evaluation_samples()
    tot_samples = len(evaluation_samples)
    if args.max_samples:
        # This will apply max_samples globally before splitting
        evaluation_samples = evaluation_samples[:args.max_samples]
    samples_for_rank = evaluation_samples[rank::args.gpus]
    print(f"Rank {rank}: processing {len(samples_for_rank)} samples on cuda:{rank} out of total {tot_samples} samples.")
    evaluator = VERLLMEvaluator(args.model_path, dataset=None, use_thinking=use_thinking, device=rank)
    _ = evaluator.evaluate_all_samples(output_dir=output_dir, samples=samples_for_rank, rank=rank)


def main():
    args = parse_args()
    
    # Determine output directory
    if args.output_dir is None:
        output_dir = derive_output_dir_from_model_path(args.model_path)
    else:
        output_dir = args.output_dir
    
    print(f"Output directory: {output_dir}")
    # Initialize evaluator / multi-GPU support
    use_thinking = not args.no_thinking
    print(f"Using thinking process: {use_thinking}")

    if args.gpus > 1:
        print(f"Spawning {args.gpus} processes for multi-GPU evaluation...")
        # spawn processes
        mp.spawn(worker_entry, args=(args, output_dir, use_thinking), nprocs=args.gpus, join=True)
        # merge per-rank results
        print("Merging per-rank results...")
        # collect and merge
        all_results = []
        all_reasoning_ious = []
        all_answer_ious = []
        per_cls_reasoning_ious = {}
        per_cls_answer_ious = {}
        per_cls_sample_count = {}
        per_cls_reasoning_mask_count = {}
        per_cls_answer_mask_count = {}

        for r in range(args.gpus):
            part_file = os.path.join(output_dir, f'ver_evaluation_results_rank{r}.json')
            if not os.path.exists(part_file):
                raise FileNotFoundError(f"Missing partial result {part_file}")
            with open(part_file, 'r') as f:
                part = json.load(f)
            all_results.extend(part.get('detailed_results', []))
            all_reasoning_ious.extend(part.get('raw_reasoning_ious', []))
            all_answer_ious.extend(part.get('raw_answer_ious', []))

            for k, v in part.get('per_cls_reasoning_ious', {}).items():
                per_cls_reasoning_ious.setdefault(k, []).extend(v)
            for k, v in part.get('per_cls_answer_ious', {}).items():
                per_cls_answer_ious.setdefault(k, []).extend(v)
            
            # Merge counts
            for k, v in part.get('per_cls_sample_count', {}).items():
                per_cls_sample_count[k] = per_cls_sample_count.get(k, 0) + v
            for k, v in part.get('per_cls_reasoning_mask_count', {}).items():
                per_cls_reasoning_mask_count[k] = per_cls_reasoning_mask_count.get(k, 0) + v
            for k, v in part.get('per_cls_answer_mask_count', {}).items():
                per_cls_answer_mask_count[k] = per_cls_answer_mask_count.get(k, 0) + v

        reasoning_ious = np.array(all_reasoning_ious)
        answer_ious = np.array(all_answer_ious)

        # recompute final metrics same as single-process case
        final_reasoning_miou = np.mean(reasoning_ious) if len(reasoning_ious) > 0 else 0.0
        final_answer_miou = np.mean(answer_ious) if len(answer_ious) > 0 else 0.0

        final_r_lq = (reasoning_ious > 0.5).sum() / len(reasoning_ious) if len(reasoning_ious) >0 else 0.0
        final_r_sq = reasoning_ious[reasoning_ious > 0.5].mean() if (reasoning_ious > 0.5).sum() >0 else 0.0

        final_a_lq = (answer_ious > 0.5).sum() / len(answer_ious) if len(answer_ious) >0 else 0.0
        final_a_sq = answer_ious[answer_ious > 0.5].mean() if (answer_ious > 0.5).sum() >0 else 0.0

        per_cls_metrics = {}
        for cls_id, ious in per_cls_reasoning_ious.items():
            ious_np = np.array(ious)
            cls_r_miou = np.mean(ious_np) if len(ious_np) > 0 else 0.0
            cls_r_lq = (ious_np > 0.5).sum() / len(ious_np) if len(ious_np) >0 else 0.0
            cls_r_sq = ious_np[ious_np > 0.5].mean() if (ious_np > 0.5).sum() >0 else 0.0

            per_cls_metrics[cls_id] = {
                'reasoning_miou': cls_r_miou,
                'reasoning_lq': cls_r_lq,
                'reasoning_sq': cls_r_sq,
                'sample_count': per_cls_sample_count.get(cls_id, 0),
                'reasoning_mask_count': per_cls_reasoning_mask_count.get(cls_id, 0),
            }

        for cls_id, ious in per_cls_answer_ious.items():
            ious_np = np.array(ious)
            cls_a_miou = np.mean(ious_np) if len(ious_np) > 0 else 0.0
            cls_a_lq = (ious_np > 0.5).sum() / len(ious_np) if len(ious_np) >0 else 0.0
            cls_a_sq = ious_np[ious_np > 0.5].mean() if (ious_np > 0.5).sum() >0 else 0.0

            per_cls_metrics.setdefault(cls_id, {}).update({
                'answer_miou': cls_a_miou,
                'answer_lq': cls_a_lq,
                'answer_sq': cls_a_sq,
                'answer_mask_count': per_cls_answer_mask_count.get(cls_id, 0),
            })

        final_summary = {
            'model_path': args.model_path,
            'use_thinking': use_thinking,
            'total_samples': len(all_results),
            'total_reasoning_masks_evaluated': len(reasoning_ious),
            'total_answer_masks_evaluated': len(answer_ious),
            'metrics': {
                'reasoning_miou': final_reasoning_miou,
                'answer_miou': final_answer_miou,
                'reasoning_lq': final_r_lq,
                'reasoning_sq': final_r_sq,
                'answer_lq': final_a_lq,
                'answer_sq': final_a_sq,
            },
            'detailed_results': all_results,
            'raw_reasoning_ious': all_reasoning_ious,
            'raw_answer_ious': all_answer_ious,
            'per_cls_reasoning_ious': per_cls_reasoning_ious,
            'per_cls_answer_ious': per_cls_answer_ious,
            'per_cls_sample_count': per_cls_sample_count,
            'per_cls_reasoning_mask_count': per_cls_reasoning_mask_count,
            'per_cls_answer_mask_count': per_cls_answer_mask_count,
            'per_cls_metrics': per_cls_metrics,
        }

        final_file = os.path.join(output_dir, 'ver_evaluation_results.json')
        with open(final_file, 'w') as f:
            json.dump(final_summary, f, indent=2)

        print(f"Merged results saved to: {final_file}")
        # Print a single consolidated summary once
        print_evaluation_summary(final_summary)

    else:
        # evaluator = VERLLMEvaluator(args.model_path, dataset, use_thinking)
        
        # # Limit samples for testing if specified
        # if args.max_samples:
        #     print(f"Limiting evaluation to {args.max_samples} samples")
        #     evaluation_samples = dataset.get_evaluation_samples()[:args.max_samples]
        #     # Temporarily override the dataset method for testing
        #     dataset.get_evaluation_samples = lambda: evaluation_samples
        
        # # Run evaluation
        # results = evaluator.evaluate_all_samples(output_dir)
        # print_evaluation_summary(results)

        pass
    
    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    print("Main sees", torch.cuda.device_count(), "GPUs")
    main()
