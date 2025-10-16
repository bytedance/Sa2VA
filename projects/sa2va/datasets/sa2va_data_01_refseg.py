import random
from typing import Literal, Optional
import torch

import numpy as np
from pycocotools import mask as mask_utils

from .common import SEG_QUESTIONS, ANSWER_LIST
from .base import Sa2VABaseDataset

from third_parts.mmdet.datasets.refcoco import RefCocoDataset

class Sa2VA01RefSeg(RefCocoDataset, Sa2VABaseDataset):

    def __init__(self,
                 data_root,
                 ann_file=None,
                 split_file=None,
                 special_tokens=None,
                 prompt_template=None,
                 extra_image_processor=None,
                 data_prefix=dict(img_path='train2014/'),
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 single_image_mode=False,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 repeats:int = 1,
                 name: str = 'Sa2VA01RefSeg',
                 **kwargs):
        
        # Initialize RefCocoDataset
        RefCocoDataset.__init__(self,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=None,
            ann_file=ann_file,
            split_file=split_file,
            **kwargs,
        )
        
        # Initialize Sa2VABaseDataset with common functionality
        Sa2VABaseDataset.__init__(self,
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            extra_image_processor=extra_image_processor,
            repeats=repeats,
            name=name
        )
        
        # Dataset-specific configurations
        self.begin_str = f'<image>\n'
        self.image_folder = data_root
        self.num_classes_per_sample = num_classes_per_sample
        self.single_image_mode = single_image_mode

    @property
    def modality_length(self):
        return [self._get_modality_length_default(100) for _ in range(len(self))]

    def _parse_annotations(self, ann_info):
        image_path = ann_info['img_path']
        image = self._read_image(image_path)
        if image is None:
            return None
        width, height = image.size

        masks, phrases = [], []
        instances, text = ann_info['instances'], ann_info['text']
        index = np.random.choice(range(len(instances)), self.num_classes_per_sample, replace=True)
        for idx in index:
            inst = instances[idx]
            phrase = text[idx].lower()
            if '.' == phrase[-1]:
                phrase = phrase[:-1]
            phrases.append(phrase)
            binary_mask = np.zeros((height, width), dtype=np.uint8)
            for seg in inst["mask"]:
                rles = mask_utils.frPyObjects([seg], height, width)
                m = mask_utils.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()
            masks.append(binary_mask)

        conversation = []
        for i, phrase in enumerate(phrases):
            question = random.choice(SEG_QUESTIONS).format(class_name=phrase)
            if i == 0:
                question = self.begin_str + question
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})
        masks = torch.stack([torch.from_numpy(mask) for mask in masks], dim=0)

        ann_info.update({
            'masks': masks,
            'conversations': conversation,
            'image': image_path
        })
        return ann_info

    def prepare_data(self, index):
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {}
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image = self._read_image(image_file)
            if image is None:
                return None
            
            # Process image using base class method
            image_data = self._process_single_image(image, self.single_image_mode)
            out_data_dict.update(image_data)
            
            # Create image token string and get input/labels
            image_token_str = self._create_image_token_string(image_data['num_image_tokens'])
            conversation = self._process_conversations_for_encoding(data_dict['conversations'], image_token_str)
            token_dict = self.get_inputid_labels(conversation)
            out_data_dict.update(token_dict)
        else:
            conversation = self._process_conversations_for_encoding(data_dict['conversations'], None)
            token_dict = self.get_inputid_labels(conversation)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(1, 3, self.image_size, self.image_size)
        return out_data_dict

    def mock_prepare_data(self, index):
        """
        Mock version of prepare_data that only checks image existence.
        Useful for testing and validation without loading full data.
        
        Returns:
            dict with status information or None if image doesn't exist
        """
        data_dict = super().prepare_data(index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        mock_data_dict = {}
        
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            if not self._check_image_exists(image_file):
                print(f'Image does not exist: {image_file}', flush=True)
                return None
            
            # Return basic information about the data without processing
            mock_data_dict.update({
                'image_path': image_file,
                'has_image': True,
                'num_masks': len(data_dict.get('masks', [])),
                'num_conversations': len(data_dict.get('conversations', [])),
                'status': 'valid'
            })
        else:
            mock_data_dict.update({
                'has_image': False,
                'num_masks': len(data_dict.get('masks', [])),
                'num_conversations': len(data_dict.get('conversations', [])),
                'status': 'valid'
            })
            
        return mock_data_dict

    def real_len(self):
        if self.serialize_data:
            return len(self.data_address)
        else:
            return len(self.data_list)

    # !DO NOT CHANGE!
    # Write the following again to override the 
    # default multi-inheritance behavior
    # Make sure the following code consistent with 
    # Sa2VABaseDataset
    # __len__
    # __getitem__
    def __len__(self):
        """Get total length considering repeats."""
        return int(self.real_len() * self.repeats)

    def __getitem__(self, index):
        """Unified __getitem__ implementation with refetch logic."""
        # Handle repeats using index mapping for equal distribution
        index_mapping = self._get_index_mapping()
        mapped_index = index_mapping[index]
        
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(mapped_index)
            # Broken images may cause the returned data to be None
            if data is None:
                mapped_index = self._rand_another_index()
                continue
            return data
        
        # If we reach here, all retries failed
        raise RuntimeError(f"Failed to get valid data after {self._max_refetch + 1} attempts")
