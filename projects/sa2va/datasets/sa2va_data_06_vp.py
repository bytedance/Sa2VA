import copy
import json
import os
import random
from typing import Literal, Dict, List, Any, Union
import torch
import numpy as np
from pycocotools import mask as mask_utils

from .base import Sa2VABaseDataset
from .common import VP_QUESTIONS


class Sa2VA06VPDataset(Sa2VABaseDataset):
    """Sa2VA implementation for Visual Prompt (VP) datasets.
    
    This dataset handles various Osprey visual prompt formats including:
    - Osprey conversation dataset
    - Osprey detailed description dataset  
    - Osprey short description dataset
    - Osprey part-level dataset
    - Osprey positive-negative dataset
    """


    def __init__(self,
                 data_path: str,
                 image_folder: Union[str, List[str]],
                 tokenizer=None,
                 prompt_template=None,
                 max_length: int = 2048,
                 special_tokens=None,
                 arch_type: Literal['intern_vl', 'qwen', 'llava'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 single_image_mode: bool = False,
                 dataset_type: str = 'conversation',  # 'conversation', 'description', 'short', 'part', 'positive_neg'
                 **kwargs):
        """
        Initialize VP dataset.
        
        Args:
            data_path: Path to the annotation file
            image_folder: Path to the image folder or list of folders
            dataset_type: Type of VP dataset format
            Other args are passed to Sa2VABaseDataset
        """
        
        # Initialize base dataset
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            **kwargs
        )
        
        # Dataset-specific configurations
        self.data_path = data_path
        self.image_folder = image_folder
        self.dataset_type = dataset_type
        self.single_image_mode = single_image_mode
        
        # VP-specific tokens
        self.VP_START_TOKEN = '<vp>'
        self.VP_END_TOKEN = '</vp>'
        self.IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        
        # Configure question limit based on dataset type
        if self.dataset_type == 'short':
            self.LIMIT = ' Answer the question using a single word or phrase.'
        else:
            self.LIMIT = ''
        
        # Load and preprocess data
        self.data_list = self._load_annotations()

    def _load_annotations(self) -> List[Dict]:
        """Load VP dataset annotations."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data

    def real_len(self) -> int:
        """Get the actual length without repeats."""
        return len(self.data_list)

    @property
    def modality_length(self):
        return [self._get_modality_length_default(100) for _ in range(len(self))]

    def _decode_masks(self, object_masks: List, height: int, width: int) -> torch.Tensor:
        """Decode RLE masks to binary masks."""
        def annToMask(mask_ann, h, w):
            if isinstance(mask_ann, list):
                rles = mask_utils.frPyObjects(mask_ann, h, w)
                rle = mask_utils.merge(rles)
            elif isinstance(mask_ann['counts'], list):
                # uncompressed RLE
                rle = mask_utils.frPyObjects(mask_ann, h, w)
            else:
                # rle
                rle = mask_ann
            mask = mask_utils.decode(rle)
            return mask

        binary_masks = []
        for object_mask in object_masks:
            binary_mask = annToMask(object_mask, height, width)
            binary_masks.append(binary_mask)
        
        if len(binary_masks) == 0:
            return None
            
        masks = np.stack(binary_masks, axis=0)
        masks = torch.from_numpy(masks)
        return masks

    def _get_region_infos(self, masks: torch.Tensor) -> tuple:
        """Get region information for VP tokens."""
        if masks is None:
            return None, []
            
        # Calculate downsample ratio and patch size based on architecture
        if self.arch_type == 'llava':
            downsample_ratio = 1.0
            image_size = 336
            patch_size = 14
        else:
            downsample_ratio = 0.5
            image_size = 448  
            patch_size = 14
            
        # Resize masks to match patch grid
        target_size = int(image_size // patch_size * downsample_ratio)
        masks_resized = torch.nn.functional.interpolate(
            masks.unsqueeze(0).float(),
            size=(target_size, target_size),
            mode='nearest'
        ).squeeze(0)
        
        # Count pixels for each region
        region_pixels = []
        for mask in masks_resized:
            region_pixels.append(int(mask.bool().sum().item()))

        # Return None if any region has fewer than 4 pixels
        if any(pixels < 4 for pixels in region_pixels):
            return None, []

        return masks_resized, region_pixels

    def _create_region_prompt(self, n_regions: int, region_pixels: List[int]) -> str:
        """Create region prompt with VP tokens."""
        start_region_str = '<image>\nThere are {} part regions in the picture: '.format(n_regions)
        for i in range(n_regions):
            start_region_str += f"region{i+1}" + self.VP_START_TOKEN + \
                               self.IMG_CONTEXT_TOKEN * region_pixels[i] + self.VP_END_TOKEN
            if i == n_regions - 1:
                start_region_str += '.\n'
            else:
                start_region_str += ', '
        return start_region_str

    def _parse_conversation_format(self, ann_info: Dict) -> Dict:
        """Parse conversation format annotations."""
        file_name = ann_info['file_name']
        conversations = ann_info['conversations']
        masks = [anno["segmentation"] for anno in ann_info["annotation"]]
        height = ann_info['height']
        width = ann_info['width']

        # Find image file path
        image_path = self._find_image_path(file_name)
        if image_path is None:
            return None

        # Decode masks
        masks = self._decode_masks(masks, height, width)
        if masks is None:
            return None
            
        masks, region_pixels = self._get_region_infos(masks)
        if masks is None:
            return None

        # Process conversations
        processed_conversations = self._process_conversations(
            conversations, len(masks), region_pixels
        )

        return {
            'image_path': image_path,
            'masks': masks,
            'conversations': processed_conversations
        }

    def _parse_description_format(self, ann_info: Dict) -> Dict:
        """Parse description format annotations."""
        file_name = ann_info['file_name']
        # Handle both 'description' and 'descriptions' keys
        descriptions = ann_info.get('description', ann_info.get('descriptions', []))
        masks = [anno["segmentation"] for anno in ann_info["annotation"]]
        height = ann_info['height']
        width = ann_info['width']

        # Find image file path
        image_path = self._find_image_path(file_name)
        if image_path is None:
            return None

        # Decode masks
        masks = self._decode_masks(masks, height, width)
        if masks is None:
            return None
            
        masks, region_pixels = self._get_region_infos(masks)
        if masks is None:
            return None

        # Create conversations from descriptions
        conversations = self._create_description_conversations(
            descriptions, len(masks), region_pixels
        )

        return {
            'image_path': image_path,
            'masks': masks,
            'conversations': conversations
        }

    def _find_image_path(self, file_name: str) -> str:
        """Find the full path to an image file."""
        if isinstance(self.image_folder, list):
            for folder in self.image_folder:
                image_path = os.path.join(folder, file_name)
                if os.path.exists(image_path):
                    return image_path
            return None
        else:
            image_path = os.path.join(self.image_folder, file_name)
            return image_path if os.path.exists(image_path) else None

    def _process_conversations(self, conversations: List[Dict], n_regions: int, region_pixels: List[int]) -> List[Dict]:
        """Process conversations with VP tokens."""
        start_region_str = self._create_region_prompt(n_regions, region_pixels)
        
        processed = []
        for i, conv in enumerate(conversations):
            processed_conv = copy.deepcopy(conv)
            # Clean HTML tags
            processed_conv['value'] = processed_conv['value'].replace('<', '').replace('>', '')
            if processed_conv['from'] == 'human':
                processed_conv['value'] += self.LIMIT
                # Add region prompt to first human message
                if i == 0:
                    processed_conv['value'] = start_region_str + processed_conv['value']
            processed.append(processed_conv)
        
        # Convert to input/output format
        messages = processed
        input_text = ''
        conversation = []
        
        # Skip initial gpt messages
        while messages and messages[0]['from'] == 'gpt':
            messages = messages[1:]
            
        for msg in messages:
            if msg['from'] == 'human':
                input_text += msg['value']
            elif msg['from'] == 'gpt':
                conversation.append({'from': 'human', 'value': input_text})
                conversation.append({'from': 'gpt', 'value': msg['value']})
                input_text = ''
                
        return conversation

    def _create_description_conversations(self, descriptions: List[str], n_regions: int, region_pixels: List[int]) -> List[Dict]:
        """Create conversations from descriptions."""
        start_region_str = self._create_region_prompt(n_regions, region_pixels)
        
        conversations = []
        for i, description in enumerate(descriptions):
            question = random.choice(VP_QUESTIONS).strip()
            question = question.replace('<region>', f"region{i+1}") + self.LIMIT
            answer = description.replace('<', '').replace('>', '')
            
            # Add region prompt to first question
            if i == 0:
                question = start_region_str + question
                
            conversations.append({'from': 'human', 'value': question})
            conversations.append({'from': 'gpt', 'value': answer})
            
        return conversations

    def prepare_data(self, index: int) -> Dict[str, Any]:
        """Prepare data for training."""
        data_dict = copy.deepcopy(self.data_list[index])
        
        # Parse annotations based on dataset type
        if self.dataset_type in ['conversation', 'part', 'positive_neg', 'short']:
            data_dict = self._parse_conversation_format(data_dict)
        elif self.dataset_type in ['description']:
            data_dict = self._parse_description_format(data_dict)
        else:
            raise ValueError(f"self.dataset_type is {self.dataset_type}")

        # Skip samples without valid data
        if data_dict is None or data_dict.get('masks') is None:
            return None

        # Process image
        image_file = data_dict['image_path']
        image = self._read_image(image_file)
        if image is None:
            return None
        
        out_data_dict = {}
        
        # Add masks
        out_data_dict['prompt_masks'] = data_dict['masks']
        
        # Process image using base class method
        image_data = self._process_single_image(image, self.single_image_mode)
        out_data_dict.update(image_data)
        
        vp_overall_mask = torch.Tensor([False] * (len(out_data_dict['pixel_values']) - 1) + [True])
        out_data_dict['vp_overall_mask'] = vp_overall_mask
        
        # Create image token string and process conversations
        image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
        conversation = self._process_conversations_for_encoding(data_dict['conversations'], image_token_str)
        token_dict = self.get_inputid_labels(conversation)
        out_data_dict.update(token_dict)
        return out_data_dict


    def mock_prepare_data(self, index: int) -> Dict[str, Any]:
        """
        Mock version of prepare_data that only checks image existence.
        Useful for testing and validation without loading full data.
        
        Returns:
            dict with status information or None if image doesn't exist
        """
        data_dict = copy.deepcopy(self.data_list[index])
        
        mock_data_dict = {}
        
        # Get file name and find image path
        file_name = data_dict['file_name']
        image_path = self._find_image_path(file_name)
        
        if image_path is None or not self._check_image_exists(image_path):
            print(f'Image does not exist: {image_path or file_name}', flush=True)
            return None
        
        # Get basic info without processing full data
        masks = data_dict.get("annotation", [])
        conversations = data_dict.get('conversations', [])
        descriptions = data_dict.get('description', data_dict.get('descriptions', []))
        
        mock_data_dict.update({
            'image_path': image_path,
            'file_name': file_name,
            'has_image': True,
            'num_masks': len(masks),
            'num_conversations': len(conversations) if conversations else len(descriptions),
            'dataset_type': self.dataset_type,
            'height': data_dict.get('height', 0),
            'width': data_dict.get('width', 0),
            'status': 'valid'
        })
        
        return mock_data_dict


# Specific dataset classes for different VP formats
class Sa2VA06OspreyDataset(Sa2VA06VPDataset):
    """Sa2VA Osprey Conversation Dataset."""
    def __init__(self, **kwargs):
        kwargs['dataset_type'] = 'conversation'
        super().__init__(**kwargs)


class Sa2VA06OspreyDescriptionDataset(Sa2VA06VPDataset):
    """Sa2VA Osprey Description Dataset."""
    def __init__(self, **kwargs):
        kwargs['dataset_type'] = 'description'
        super().__init__(**kwargs)


class Sa2VA06OspreyShortDescriptionDataset(Sa2VA06VPDataset):
    """Sa2VA Osprey Short Description Dataset."""
    def __init__(self, **kwargs):
        kwargs['dataset_type'] = 'short'
        super().__init__(**kwargs)


class Sa2VA06OspreyPartDataset(Sa2VA06VPDataset):
    """Sa2VA Osprey Part-level Dataset."""
    def __init__(self, **kwargs):
        kwargs['dataset_type'] = 'part'
        super().__init__(**kwargs)


class Sa2VA06OspreyPositiveNegDataset(Sa2VA06VPDataset):
    """Sa2VA Osprey Positive-Negative Dataset."""
    def __init__(self, **kwargs):
        kwargs['dataset_type'] = 'positive_neg'
        super().__init__(**kwargs)
