"""
Second-version VER SA1B dataloader based on Sa2VA unified architecture.

This version:
- Inherits from Sa2VABaseDataset for unified model support
- Supports TFRecord and JSON file loading
- Implements thinking mode for visual evidence reasoning
- Supports GRPO training (without language labels)
- Supports key filtering for subset training
"""

import copy
import io
import json
import random
import glob
import os
from typing import Dict, Literal, Optional, List

import torch
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from tfd_utils import TFRecordRandomAccess

from xtuner.utils import IGNORE_INDEX
from projects.sa2va.datasets.base import Sa2VABaseDataset
from projects.vrt_sa2va.data_loader.utils import (
    replace_ver_tags_with_nothing, 
    rle_to_mask, 
    extract_tagged_numbers_keep_original_tag, 
    replace_tagged_objects_with_special_token
)

DEFAULT_SA1B_DIR = 'data/SA-1B/*.tfrecord'

# Think prompt templates for VER
VER_THINK_PROMPT = [
    'You should first think about the reasoning process in the mind and then provides the user with the answer. Please respond with segmentation mask in both the thinking process and the answer.',
    'You should first think about the reasoning process in the mind and then provides the user with the answer. Please output the segmentation mask in both the thinking process and the answer.',
]

# SFT and GRPO prompt
THINK_TEMPLATE_PROMPT = [
    'The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answers here </answer>.'
]

# think answer template
THINK_ANSWER_PROMPT = [
    '<think>{reasoning}</think>\n\n<answer>{answer}</answer>'
]


class Sa2VAVRTTrain(Sa2VABaseDataset):
    """Second-version Dataset for VRT SA1B data with Sa2VA unified architecture.
    
    This dataset loader is based on the unified Sa2VA architecture (projects/sa2va)
    instead of the older llava_sam2 architecture. It provides better model support
    and follows the modern dataset design patterns.
    
    Supports loading from either:
    1. Individual JSON files (original behavior) - set json_dir parameter
    2. TFRecord files created by build_training_data.py - set tfrecord_pattern parameter
    
    Args:
        json_dir (str, optional): Directory containing individual JSON annotation files
        tfrecord_pattern (str, optional): Pattern for TFRecord files with training data
        sa1b_dirs (Dict): Dictionary mapping SA1B TFRecord file paths (only needed for JSON mode)
        thinking_mode (bool, default=True): Whether to enable thinking mode for VER training.
            When True, uses think prompts and formats answers with <think></think> and <answer></answer> tags.
            When False, uses direct question-answer format without thinking prompts.
        with_reasoning_mask (bool, default=True): Whether to include masks in reasoning section.
            When False, removes <ver>...</ver> tags from reasoning.
        with_language_labels (bool, default=True): Whether to include language output labels.
            When False, only provides question without answer (for GRPO training).
        keys_json_file (str, optional): Path to JSON file containing specific keys to filter from dataset.
            When provided, only samples with keys listed in this file will be used.
        single_image_mode (bool, default=False): Whether to use single image mode (no dynamic preprocessing)
        repeats (float, default=1.0): Number of times to repeat the dataset
        name (str): Name of the dataset for logging
        ... (other Sa2VABaseDataset parameters)
    """
    
    def __init__(self,
                 # Data source parameters
                 json_dir: Optional[str] = None,
                 tfrecord_pattern: Optional[str] = None,
                 sa1b_dirs: Dict = {},
                 
                 # VER-specific parameters
                 thinking_mode: bool = True,
                 with_reasoning_mask: bool = True,
                 with_language_labels: bool = True,
                 keys_json_file: Optional[str] = None,
                 
                 # Subset parameters
                 subset_dict: Optional[Dict] = None,
                 
                 # Dataset mode parameters
                 single_image_mode: bool = False,
                 
                 # Sa2VABaseDataset parameters
                 tokenizer=None,
                 prompt_template=None,
                 max_length: int = 2048,
                 special_tokens: Optional[List[str]] = None,
                 arch_type: Literal['intern_vl', 'qwen', 'llava'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 repeats: float = 1.0,
                 name: str = 'Sa2VAVERTrain',
                 **kwargs):
        
        # Initialize base dataset
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name,
            **kwargs
        )
        
        # VER-specific configurations
        self.thinking_mode = thinking_mode
        self.with_reasoning_mask = with_reasoning_mask
        self.with_language_labels = with_language_labels
        self.single_image_mode = single_image_mode
        
        # Setup subset configuration
        if subset_dict is None:
            subset_dict = {'mode': 'all', 'num': 0}
        self.subset_dict = subset_dict
        
        # Load keys from JSON file if provided
        self.filter_keys = None
        if keys_json_file is not None:
            assert os.path.exists(keys_json_file), f"Keys JSON file does not exist: {keys_json_file}"
            with open(keys_json_file, 'r') as f:
                keys_data = json.load(f)['samples']
            
            # Handle different JSON formats
            if isinstance(keys_data, list):
                self.filter_keys = set(keys_data)
            elif isinstance(keys_data, dict):
                self.filter_keys = set(keys_data.keys())
            else:
                raise ValueError(f"Keys JSON file must contain a list or dictionary, got {type(keys_data)}")
            
            print(f"Loaded {len(self.filter_keys)} keys from {keys_json_file}")
        
        # Support both JSON files and TFRecord files
        self.use_tfrecord = tfrecord_pattern is not None
        self.datas = []
        
        if self.use_tfrecord:
            # Initialize TFRecord reader
            self.tfrecord_reader = TFRecordRandomAccess(tfrecord_pattern, key_feature_name='key')
            
            # Get all keys from TFRecord
            print("Loading from TFRecord files...")
            all_keys = list(self.tfrecord_reader.get_keys())
            print(f"Found {len(all_keys)} keys in TFRecord files")
            
            if not all_keys:
                raise ValueError(f"No keys found in TFRecord files: {tfrecord_pattern}")
            
            # Filter keys if keys_json_file is provided
            if self.filter_keys is not None:
                missing_keys = self.filter_keys - set(all_keys)
                if missing_keys:
                    raise ValueError(f"Keys from {keys_json_file} not found in dataset: {missing_keys}")
                
                all_keys = [key for key in all_keys if key in self.filter_keys]
                print(f"Filtered to {len(all_keys)} keys specified in {keys_json_file}")
            
            # Apply subset filtering
            if subset_dict['mode'] == 'first':
                all_keys = all_keys[:subset_dict['num']]
            elif subset_dict['mode'] == 'random':
                random.shuffle(all_keys)
                all_keys = all_keys[:subset_dict['num']]
            elif subset_dict['mode'] == 'all':
                pass
            else:
                raise ValueError(f"Unknown subset mode: {subset_dict['mode']}")
            
            self.datas = all_keys
            print(f"Loaded {len(self.datas)} samples from TFRecord files")
        else:
            # Load from JSON files (original behavior)
            assert json_dir is not None, "Either json_dir or tfrecord_pattern must be provided"
            json_files = glob.glob(os.path.join(json_dir, "*.json"))
            
            if self.filter_keys is not None:
                raise NotImplementedError("Filtering by keys is only supported in TFRecord mode currently.")
            
            # Apply subset filtering
            if subset_dict['mode'] == 'first':
                json_files = json_files[:subset_dict['num']]
            elif subset_dict['mode'] == 'random':
                random.shuffle(json_files)
                json_files = json_files[:subset_dict['num']]
            elif subset_dict['mode'] == 'all':
                pass
            else:
                raise ValueError(f"Unknown subset mode: {subset_dict['mode']}")
            
            for json_file in json_files:
                assert os.path.exists(json_file), f"JSON file does not exist: {json_file}"
                self.datas.append(json_file)
        
        # Initialize SA1B TFRecord readers (only needed when not using TFRecord mode)
        if not self.use_tfrecord:
            if sa1b_dirs is None:
                self.sa1b_dirs = {'default': DEFAULT_SA1B_DIR}
            else:
                self.sa1b_dirs = sa1b_dirs
            
            for key in self.sa1b_dirs:
                base_dir = self.sa1b_dirs[key]
                if base_dir.endswith('.tfrecord'):
                    assert os.path.exists(os.path.dirname(base_dir)), \
                        f"TFRecord directory does not exist: {base_dir}"
                    file_handler = TFRecordRandomAccess(base_dir, key_feature_name='key')
                    self.sa1b_dirs[key] = file_handler
        
        # Initialize thinking prompts only when thinking_mode is enabled
        if self.thinking_mode:
            self.think_prompt = VER_THINK_PROMPT
            self.think_template_prompt = THINK_TEMPLATE_PROMPT
            self.think_answer_prompt = THINK_ANSWER_PROMPT
        else:
            self.think_prompt = None
            self.think_template_prompt = None
            self.think_answer_prompt = None
    
    def real_len(self):
        """Get the actual length without repeats."""
        return len(self.datas)
    
    @property
    def modality_length(self):
        """Get modality length for all items."""
        return [self._get_modality_length_default(100) for _ in range(len(self))]
    
    @staticmethod
    def get_a_example_prompt(question: str, reasoning: bool = True) -> str:
        """Get an example prompt for Sa2VA VER.
        
        Args:
            question: The question text
            reasoning: Whether to include reasoning prompts
            
        Returns:
            Formatted question with image token and reasoning prompts
        """
        if reasoning:
            think_prompt = VER_THINK_PROMPT[0]
            question_with_think = f"{question}\n\n{think_prompt}"
            question = f"{question_with_think}\n\n{THINK_TEMPLATE_PROMPT[0]}"
        question = f'<image>\n{question}'
        return question
    
    def _get_image(self, image_path: str) -> Image.Image:
        """Get image from SA1B dataset (JSON mode only).
        
        Args:
            image_path: Path like "sa_000000/sa_1000.jpg"
            
        Returns:
            PIL Image
        """
        image_file_name = image_path.split('/')[-1]
        image_key = image_file_name.split('.')[0]  # e.g., "sa_1000"
        
        image_bytes = self.sa1b_dirs['default'].get_feature(image_key, feature_name='image')
        if image_bytes is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return img
    
    def _conv_to_input_ids(self, conversations: List[Dict], image_token_str: str, 
                          with_labels: bool = True) -> Dict:
        """Convert conversations to input_ids and labels.
        
        This is the original implementation from ver_sa1b.py that supports with_labels=False.
        Used instead of base class method to support GRPO training.
        
        Args:
            conversations: List of conversation dictionaries
            image_token_str: String representation of image tokens
            with_labels: Whether to include labels (False for GRPO training)
            
        Returns:
            Dictionary with input_ids and labels
        """
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]
        for msg in conversations:
            if msg['from'] == 'human':
                if image_token_str is None and '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', '')
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>', image_token_str).strip()
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)

            if with_labels:
                output_text = single_turn_conversation.get('output', '')
                if self.template.get('SUFFIX', None):
                    output_text += self.template.SUFFIX
                output_encode = self.tokenizer.encode(
                    output_text, add_special_tokens=False)
                input_ids += output_encode
                labels += copy.deepcopy(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]

        return {'input_ids': input_ids, 'labels': labels}
    
    def _parse_annotations(self, ann_info: Dict, key: Optional[str] = None, 
                          image_bytes: Optional[bytes] = None) -> Optional[Dict]:
        """Parse VER annotations to training format.
        
        Args:
            ann_info: Annotation dictionary
            key: Data key (for TFRecord mode)
            image_bytes: Image bytes (for TFRecord mode)
            
        Returns:
            Processed annotation dictionary or None if invalid
        """
        # Get the VER data from generated candidates
        ver_data = ann_info['generated_ver_data']
        if not ver_data['candidates']:
            return None
        
        # Take the first candidate (highest scored)
        candidate = ver_data['candidates'][0]
        
        question = candidate['question']
        reasoning = candidate['reasoning']
        answer = candidate['answer_caption']
        
        # Handle thinking mode
        if self.thinking_mode and self.think_prompt is not None and \
           self.think_template_prompt is not None and self.think_answer_prompt is not None:
            # Add think prompt
            think_prompt = random.choice(self.think_prompt)
            question_with_think = f"{question}\n\n{think_prompt}"
            
            # Add template prompt
            question = f"{question_with_think}\n\n{self.think_template_prompt[0]}"
            question = f'<image>\n{question}'
            
            if not self.with_reasoning_mask:
                # Remove any <ver>...</ver> segments from reasoning and answer
                reasoning = replace_ver_tags_with_nothing(reasoning)
            
            # Format answer with thinking template
            answer_formatted = self.think_answer_prompt[0].format(reasoning=reasoning, answer=answer)
        else:
            # Direct question without thinking prompts
            question = f'<image>\n{question}'
            # Direct answer without thinking format
            answer_formatted = answer
            answer_formatted = answer_formatted.replace('<vea>', '').replace('</vea>', '')
        
        # Get image
        if image_bytes is not None:
            # Load image from provided bytes (TFRecord mode)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        else:
            # Load image using original method (JSON mode)
            image = self._get_image(ann_info['image_name'])
        
        # Process masks from original annotation
        original_ann = ann_info['original_annotation']
        objects_anns = original_ann['objects_anns']
        
        # Handle case without language labels (GRPO training)
        if not self.with_language_labels:
            answer_objects = candidate['answer_objects']
            if not answer_objects:
                return None
            
            conversations = []
            conversations.append({'from': 'human', 'value': question})
            conversations.append({'from': 'gpt', 'value': ''})
            
            masks_list = []
            for original_tag in answer_objects:
                if original_tag not in objects_anns:
                    print(f"Warning: Mask index {original_tag} not found in objects annotations.")
                    return None
                segmentation = objects_anns[original_tag]['segmentation']
                if isinstance(segmentation, list):
                    mask = rle_to_mask(segmentation)
                else:
                    mask = mask_utils.decode(segmentation)
                mask = np.uint8(mask)
                masks_list.append(mask)
            
            if not masks_list:
                return None
            
            masks = np.stack(masks_list, axis=0)
            masks = torch.from_numpy(masks)
            
            ann_info_processed = {
                'image': image,
                'masks': masks,
                'conversations': conversations,
            }
            return ann_info_processed
        
        # Extract all object tags from the answer_formatted text
        mask_indices = extract_tagged_numbers_keep_original_tag(answer_formatted)
        
        # Create mapping from original indices to new indices
        index_mapping = {}
        index_cnt = 1
        for original_tag in mask_indices:
            if original_tag not in index_mapping:
                index_mapping[original_tag] = f'<obj{index_cnt}>'
                index_cnt += 1
        
        # Reorder the text to have objects in numeric order
        # Use temporary placeholders to avoid overlapping replacements
        answer_formatted_reordered = answer_formatted
        temp_mapping = {}
        
        # First pass: replace with unique temporary placeholders
        for i, (original_tag, new_tag) in enumerate(index_mapping.items()):
            temp_placeholder = f"__TEMP_PLACEHOLDER_{i}__"
            answer_formatted_reordered = answer_formatted_reordered.replace(original_tag, temp_placeholder)
            temp_mapping[temp_placeholder] = new_tag
        
        # Second pass: replace temporary placeholders with final tags
        for temp_placeholder, new_tag in temp_mapping.items():
            answer_formatted_reordered = answer_formatted_reordered.replace(temp_placeholder, new_tag)
        
        # Process masks in the new order
        masks_list = []
        for original_tag in mask_indices:
            if original_tag in objects_anns:
                segmentation = objects_anns[original_tag]['segmentation']
                if isinstance(segmentation, list):
                    mask = rle_to_mask(segmentation)
                else:
                    mask = mask_utils.decode(segmentation)
                mask = np.uint8(mask)
                masks_list.append(mask)
            else:
                print(f"Warning: Mask index {original_tag} not found in objects annotations.")
                return None
        
        if not masks_list:
            return None
        
        masks = np.stack(masks_list, axis=0)
        masks = torch.from_numpy(masks)
        
        # Replace object tags with [SEG] tokens
        assert '[SEG]' not in answer_formatted_reordered, \
            f"[SEG] should not be in the original answer: {answer_formatted_reordered}"
        answer_formatted_final = replace_tagged_objects_with_special_token(
            answer_formatted_reordered, '[SEG]')
        
        # Create conversations
        conversations = []
        conversations.append({'from': 'human', 'value': question})
        conversations.append({'from': 'gpt', 'value': answer_formatted_final})
        
        ann_info_processed = {
            'image': image,
            'masks': masks,
            'conversations': conversations,
        }
        return ann_info_processed
    
    def prepare_data(self, idx: int) -> Optional[Dict]:
        """Prepare data for a given index.
        
        Args:
            idx: Data index
            
        Returns:
            Processed data dictionary or None if invalid
        """
        if self.use_tfrecord:
            # Load from TFRecord
            key = self.datas[idx]
            
            # Get JSON data from TFRecord
            json_bytes = self.tfrecord_reader.get_feature(key, feature_name='json')
            if json_bytes is None:
                raise ValueError(f"JSON data not found for key: {key}")
            data_dict = json.loads(json_bytes.decode('utf-8'))
            
            # Get image bytes from TFRecord
            image_bytes = self.tfrecord_reader.get_feature(key, feature_name='image')
            if image_bytes is None:
                raise ValueError(f"Image data not found for key: {key}")
            
            # Parse annotations with image bytes
            data_dict = self._parse_annotations(data_dict, key=key, image_bytes=image_bytes)
        else:
            # Load from JSON file (original behavior)
            data_dict_path = self.datas[idx]
            with open(data_dict_path, 'r') as f:
                data_dict = json.load(f)
            data_dict = self._parse_annotations(data_dict)
        
        if data_dict is None:
            return None
        
        # Process the data using base class methods
        out_data_dict = {}
        
        # Add masks if present
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']
        
        # Process image
        if 'image' in data_dict and data_dict['image'] is not None:
            image = data_dict['image']
            
            # Process image using base class method
            image_data = self._process_single_image(image, self.single_image_mode)
            out_data_dict.update(image_data)
            
            # Create image token string
            image_token_str = self._create_image_token_string(image_data['num_image_tokens'])

            if self.with_language_labels is False:
                token_dict = self._conv_to_input_ids(
                data_dict['conversations'], 
                image_token_str, 
                with_labels=False
            )
            else:
                conversations = self._process_conversations_for_encoding(
                    data_dict['conversations'], image_token_str)
                token_dict = self.get_inputid_labels(conversations)
            out_data_dict.update(token_dict)
        else:
            raise ValueError("Image is required in data_dict for VER training")
        
        return out_data_dict


if __name__ == '__main__':
    # Example usage and testing
    from transformers import AutoTokenizer
    from xtuner.utils import PROMPT_TEMPLATE
    
    # Configuration
    path = './pretrained/internvl2_5/InternVL2_5-8B'
    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
        padding_side='right')
    
    # Special tokens for segmentation
    special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']
    
    # Image processor for grounding
    from projects.sa2va.models.preprocess.image_resize import DirectResize
    extra_image_processor = dict(
        type=DirectResize,
        target_length=1024,
    )
    
    # Prompt template
    prompt_template = PROMPT_TEMPLATE.qwen_chat
    max_length = 8192
    
    # Example 1: Loading from TFRecord files with thinking mode enabled (default)
    print("=" * 80)
    print("Example 1: TFRecord with thinking mode")
    print("=" * 80)
    dataset_tfrecord = Sa2VAVRTTrain(
        tfrecord_pattern='data/ver_0801/ver_training_0000*.tfrecord',
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        prompt_template=prompt_template,
        max_length=max_length,
        arch_type='intern_vl',
        thinking_mode=True,
        name='VER_SA1B_Thinking'
    )
    print(f"Dataset length: {len(dataset_tfrecord)}")
    print(f"Real length: {dataset_tfrecord.real_len()}")
    
    # Example 2: Loading from TFRecord files with thinking mode disabled
    print("\n" + "=" * 80)
    print("Example 2: TFRecord without thinking mode")
    print("=" * 80)
    dataset_tfrecord_no_thinking = Sa2VAVRTTrain(
        tfrecord_pattern='data/ver_0801/ver_training_0000*.tfrecord',
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        prompt_template=prompt_template,
        max_length=max_length,
        arch_type='intern_vl',
        thinking_mode=False,
        subset_dict={'mode': 'first', 'num': 100},
        name='VER_SA1B_NoThinking'
    )
    print(f"Dataset length: {len(dataset_tfrecord_no_thinking)}")
    
    # Example 3: Loading without language labels (GRPO training)
    print("\n" + "=" * 80)
    print("Example 3: TFRecord for GRPO training (no language labels)")
    print("=" * 80)
    dataset_tfrecord_no_language = Sa2VAVRTTrain(
        tfrecord_pattern='data/ver_0801/ver_training_0000*.tfrecord',
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        prompt_template=prompt_template,
        max_length=max_length,
        arch_type='intern_vl',
        thinking_mode=True,
        with_language_labels=False,
        subset_dict={'mode': 'first', 'num': 100},
        name='VER_SA1B_GRPO'
    )
    print(f"Dataset length: {len(dataset_tfrecord_no_language)}")
    
    # Example 4: Loading with specific keys from JSON file
    print("\n" + "=" * 80)
    print("Example 4: TFRecord with key filtering")
    print("=" * 80)
    dataset_with_keys_filter = Sa2VAVRTTrain(
        tfrecord_pattern='data/ver_0801/ver_training_*.tfrecord',
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        prompt_template=prompt_template,
        max_length=max_length,
        arch_type='intern_vl',
        thinking_mode=True,
        with_language_labels=False,
        keys_json_file='work_dirs/group_inference_sa2va_4b_0801_grpo_ver_1k_0801_80k_output/top_2k_high_variance_samples.json',
        name='VER_SA1B_Filtered'
    )
    print(f"Dataset with keys filter loaded {len(dataset_with_keys_filter)} samples")
    
    # Test loading a sample
    print("\n" + "=" * 80)
    print("Testing sample loading...")
    print("=" * 80)
    try:
        sample = dataset_tfrecord[0]
        print(f"Sample keys: {sample.keys()}")
        if 'masks' in sample:
            print(f"Masks shape: {sample['masks'].shape}")
        if 'input_ids' in sample:
            print(f"Input IDs length: {len(sample['input_ids'])}")
        print("Sample loaded successfully!")
    except Exception as e:
        print(f"Error loading sample: {e}")
        import traceback
        traceback.print_exc()
