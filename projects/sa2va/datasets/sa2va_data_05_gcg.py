import copy
import json
import os
import random
from typing import Literal, Dict, List, Any
import torch
import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

from .base import Sa2VABaseDataset
from .common import GCG_QUESTIONS


class Sa2VA05GCGDataset(Sa2VABaseDataset):
    """Sa2VA implementation for Grounded Conversation Generation (GCG) datasets.
    
    This dataset handles various GCG formats including:
    - RefCOCOg GCG 
    - GranDf GCG
    - Flickr30k GCG
    - OpenPSG GCG
    """

    def __init__(self,
                 data_path: str,
                 image_folder: str,
                 tokenizer=None,
                 prompt_template=None,
                 max_length: int = 2048,
                 special_tokens=None,
                 arch_type: Literal['intern_vl', 'qwen', 'llava'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 single_image_mode: bool = False,
                 dataset_type: str = 'generic',  # 'refcocog', 'grandf', 'flickr30k', 'openpsg', 'generic'
                 **kwargs):
        """
        Initialize GCG dataset.
        
        Args:
            data_path: Path to the annotation file
            image_folder: Path to the image folder
            dataset_type: Type of GCG dataset format
            num_classes_per_sample: Number of classes per sample (for some formats)
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
        self.dataset_type = dataset_type.lower()
        self.single_image_mode = single_image_mode
        self.question_templates = GCG_QUESTIONS
        self.begin_str = f'<image>\n'
        
        # Load and preprocess data based on dataset type
        self.data_list = self._load_annotations()

    def _load_annotations(self) -> List[Dict]:
        """Load annotations based on dataset type."""
        if self.dataset_type == 'refcocog':
            return self._load_refcocog_annotations()
        elif self.dataset_type == 'flickr30k':
            return self._load_flickr30k_annotations()
        else:
            # Generic GCG format (used for grandf, openpsg, etc.)
            return self._load_generic_annotations()
    
    def _load_generic_annotations(self) -> List[Dict]:
        """Load generic GCG format annotations."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return data
    
    def _load_refcocog_annotations(self) -> List[Dict]:
        """Load RefCOCOg GCG format annotations."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        # RefCOCOg format has nested structure
        return [list(line.values())[0] for line in data]
    
    def _load_flickr30k_annotations(self) -> List[Dict]:
        """Load Flickr30k GCG format annotations (COCO-style)."""
        def filter_images(data_infos, min_size=32):
            return [i for i, info in enumerate(data_infos) if min(info['width'], info['height']) >= min_size]
        
        self.coco = COCO(self.data_path)
        self.image_ids = self.coco.getImgIds()
        data_infos = []
        total_ann_ids = []
        removed_img_count = 0
        
        for img_id in self.image_ids:
            info = self.coco.loadImgs([img_id])[0]
            if len(info['caption'].split(' ')) < 3:
                removed_img_count += 1
                continue
            info['filename'] = info['file_name'].split('_')[-1]
            info['height'] = int(info['height'])
            info['width'] = int(info['width'])
            data_infos.append(info)
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            total_ann_ids.extend(ann_ids)
            
        assert len(set(total_ann_ids)) == len(total_ann_ids), f"Non-unique annotation IDs in '{self.data_path}'!"
        print(f'Removed {removed_img_count} images from Flickr30k.')
        data_infos = [data_infos[i] for i in filter_images(data_infos, min_size=32)]
        
        return data_infos

    def real_len(self) -> int:
        """Get the actual length without repeats."""
        return len(self.data_list)

    @property
    def modality_length(self):
        return [self._get_modality_length_default(100) for _ in range(len(self))]

    def _parse_generic_annotations(self, ann_info: Dict) -> Dict:
        """Parse generic GCG format annotations."""
        image_path = os.path.join(self.image_folder, ann_info['file_name'])
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        caption = ann_info['caption'].strip('"').strip()
        masks, phrases, tokens_positive = [], [], []
        
        for word, grounding in ann_info.get("groundings", {}).items():
            phrases.append(word)
            tokens_positive.append(grounding["token_positives"])

            # Convert segmentation to binary mask
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for rle in grounding["rle_masks"]:
                m = mask_utils.decode(rle).astype(np.uint8)
                binary_mask += m.squeeze()
            masks.append(binary_mask)

        # Sort by token position
        if tokens_positive:
            phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
            masks = [masks[i] for i in phrase_order]
            phrases = [phrases[i] for i in phrase_order]
            tokens_positive = [tokens_positive[i] for i in phrase_order]

        ann_info.update({
            'image_path': image_path,
            'caption': caption,
            'masks': masks,
            'phrases': phrases,
            'tokens_positive': tokens_positive,
        })
        return ann_info

    def _parse_refcocog_annotations(self, ann_info: Dict) -> Dict:
        """Parse RefCOCOg GCG format annotations."""
        image_path = os.path.join(self.image_folder, ann_info['img_file_name'])
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        caption = ann_info['caption'].strip('"').strip().lower()
        masks, phrases, tokens_positive = [], [], []
        
        for detail in ann_info.get('refs', []):
            phrase = detail['sentence']
            if phrase.lower() in caption:
                phrases.append(phrase)
                index = caption.find(phrase.lower())
                end_index = index + len(phrase) if index != -1 else -1
                tokens_positive.append([index, end_index])

                binary_mask = np.zeros((height, width), dtype=np.uint8)
                for seg in detail["segmentation"]:
                    rles = mask_utils.frPyObjects([seg], height, width)
                    m = mask_utils.decode(rles)
                    m = m.astype(np.uint8)
                    binary_mask += m.squeeze()
                masks.append(binary_mask)

        # Sort by token position
        if tokens_positive:
            phrase_order = sorted(range(len(tokens_positive)), key=lambda x: tokens_positive[x][0])
            masks = [masks[i] for i in phrase_order]
            phrases = [phrases[i] for i in phrase_order]
            tokens_positive = [tokens_positive[i] for i in phrase_order]

        ann_info.update({
            'image_path': image_path,
            'caption': caption,
            'masks': masks,
            'phrases': phrases,
            'tokens_positive': tokens_positive,
        })
        return ann_info

    def _parse_flickr30k_annotations(self, img_info: Dict) -> Dict:
        """Parse Flickr30k GCG format annotations."""
        ann_ids = self.coco.getAnnIds(imgIds=img_info['id'])
        ann_info = self.coco.loadAnns(ann_ids)
        
        image_path = os.path.join(self.image_folder, img_info['file_name'])
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        annotations = {
            'phrases': [], 
            'caption': img_info['caption'], 
            'masks': [], 
            'tokens_positive': []
        }

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0 or ann['area'] <= 0 or w < 1 or h < 1:
                continue
                
            tokens_positive = ann['tokens_positive']
            phrase = [img_info['caption'][span[0]:span[1]] for span in tokens_positive]
            annotations['phrases'].append(phrase[0])
            annotations['tokens_positive'].append(tokens_positive[0])

            # Decode mask
            rle = ann['sam_mask']
            mask_decoded = mask_utils.decode(rle).astype(np.uint8)
            annotations['masks'].append(mask_decoded)

        # Sort by token position
        if annotations['tokens_positive']:
            phrase_order = sorted(range(len(annotations['tokens_positive'])), 
                                key=lambda x: annotations['tokens_positive'][x][0])
            annotations['masks'] = [annotations['masks'][i] for i in phrase_order]
            annotations['phrases'] = [annotations['phrases'][i] for i in phrase_order]
            annotations['tokens_positive'] = [annotations['tokens_positive'][i] for i in phrase_order]

        annotations.update({
            'image_path': image_path,
            'caption': annotations['caption'],
        })
        return annotations

    def _create_gcg_conversation(self, caption: str, tokens_positive: List[List[int]]) -> List[Dict]:
        """Create conversation with interleaved segmentation masks."""
        question = random.choice(self.question_templates).strip()
        question = self.begin_str + question

        # Tag caption with segmentation markers
        def tag_caption(caption, tokens):
            for start, end in sorted(tokens, key=lambda x: x[0], reverse=True):
                caption = f"{caption[:start]}<p> {caption[start:end]} </p> [SEG]{caption[end:]}"
            return caption

        detailed_answer = tag_caption(caption, tokens_positive)

        conversation = [
            {'from': 'human', 'value': question},
            {'from': 'gpt', 'value': detailed_answer}
        ]
        return conversation

    def prepare_data(self, index: int) -> Dict[str, Any]:
        """Prepare data for training."""
        data_dict = copy.deepcopy(self.data_list[index])
        
        # Parse annotations based on dataset type
        if self.dataset_type == 'refcocog':
            data_dict = self._parse_refcocog_annotations(data_dict)
        elif self.dataset_type == 'flickr30k':
            data_dict = self._parse_flickr30k_annotations(data_dict)
        else:
            data_dict = self._parse_generic_annotations(data_dict)
        
        # Skip samples without masks
        if not data_dict.get('masks') or len(data_dict['masks']) == 0:
            return None
        
        out_data_dict = {}
        
        # Process masks
        masks = torch.stack([torch.from_numpy(mask) for mask in data_dict['masks']], dim=0)
        out_data_dict['masks'] = masks
        
        # Create conversation
        conversation = self._create_gcg_conversation(
            data_dict['caption'], 
            data_dict['tokens_positive']
        )
        
        # Process image
        image_file = data_dict['image_path']
        image = self._read_image(image_file)
        if image is None:
            return None
        
        # Process image using base class method
        image_data = self._process_single_image(image, self.single_image_mode)
        out_data_dict.update(image_data)
        
        # Create image token string and process conversations
        image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
        conversation = self._process_conversations_for_encoding(conversation, image_token_str)
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
        
        # Parse basic annotation info to get image path
        if self.dataset_type == 'refcocog':
            image_path = os.path.join(self.image_folder, data_dict['img_file_name'])
            caption = data_dict.get('caption', '')
            num_refs = len(data_dict.get('refs', []))
        elif self.dataset_type == 'flickr30k':
            image_path = os.path.join(self.image_folder, data_dict['file_name'])
            caption = data_dict.get('caption', '')
            # For flickr, we need to get annotations
            ann_ids = self.coco.getAnnIds(imgIds=data_dict['id'])
            num_refs = len(ann_ids)
        else:
            # Generic format
            image_path = os.path.join(self.image_folder, data_dict['file_name'])
            caption = data_dict.get('caption', '')
            num_refs = len(data_dict.get('groundings', {}))
        
        # Check if image exists
        if not self._check_image_exists(image_path):
            print(f'Image does not exist: {image_path}', flush=True)
            return None
        
        mock_data_dict = {
            'image_path': image_path,
            'has_image': True,
            'caption_length': len(caption.split()) if caption else 0,
            'num_regions': num_refs,
            'dataset_type': self.dataset_type,
            'status': 'valid'
        }
        
        return mock_data_dict


# Specific dataset classes for different GCG formats
class Sa2VA05RefCOCOgGCGDataset(Sa2VA05GCGDataset):
    """Sa2VA RefCOCOg GCG Dataset."""
    def __init__(self, **kwargs):
        kwargs['dataset_type'] = 'refcocog'
        super().__init__(**kwargs)


class Sa2VA05GranDfGCGDataset(Sa2VA05GCGDataset):
    """Sa2VA GranDf GCG Dataset."""
    def __init__(self, **kwargs):
        kwargs['dataset_type'] = 'grandf'
        super().__init__(**kwargs)


class Sa2VA05FlickrGCGDataset(Sa2VA05GCGDataset):
    """Sa2VA Flickr30k GCG Dataset."""
    def __init__(self, **kwargs):
        kwargs['dataset_type'] = 'flickr30k'
        super().__init__(**kwargs)


class Sa2VA05OpenPsgGCGDataset(Sa2VA05GCGDataset):
    """Sa2VA OpenPSG GCG Dataset."""
    def __init__(self, **kwargs):
        kwargs['dataset_type'] = 'openpsg'
        super().__init__(**kwargs)

