import json
import os
from typing import Literal

import torch
from .base import Sa2VABaseDataset


class LLaVADataset(Sa2VABaseDataset):

    def __init__(self,
                 tokenizer,
                 data_path,
                 prompt_template=None,
                 special_tokens=None,
                 image_folder=None,
                 max_length=8192,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 skip_pure_text=False,
                 **kwargs):

        # Initialize base class
        super().__init__(
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            max_length=max_length,
            special_tokens=special_tokens,
            arch_type=arch_type,
            preprocessor=preprocessor,
            **kwargs
        )

        # Dataset-specific configurations
        self.image_folder = image_folder
        self.skip_pure_text = skip_pure_text
        self.data = self._load_annotations(data_path, image_folder)

    def _load_annotations(self, data_path, image_folder=None):
        data = json.load(open(data_path))
        return data

    def prepare_data(self, index):
        data_dict: dict = self.data[index]
        
        if data_dict is None:
            return None
        
        out_data_dict = {}

        if self.skip_pure_text and data_dict.get('image', None) is None:
            return None

        if data_dict.get('image', None) is not None:
            image_file = os.path.join(self.image_folder, data_dict['image'])
            image = self._read_image(image_file)
            if image is None:
                return None
            
            # Process image using base class method
            # For LLaVA, typically use single image mode
            single_image_mode = True if self.preprocessor is not None else False
            image_data = self._process_single_image(image, single_image_mode)
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

    def real_len(self):
        """Get the actual length without repeats."""
        return len(self.data)

    @property
    def modality_length(self):
        return [self._get_modality_length_default(100) for _ in range(len(self))]

    def mock_prepare_data(self, index):
        """
        Mock version of prepare_data that only checks image existence.
        Useful for testing and validation without loading full data.
        
        Returns:
            dict with status information or None if image doesn't exist
        """
        data_dict: dict = self.data[index]
        
        if data_dict is None:
            return None
        
        mock_data_dict = {}

        if self.skip_pure_text and data_dict.get('image', None) is None:
            return None

        if data_dict.get('image', None) is not None:
            image_file = os.path.join(self.image_folder, data_dict['image'])
            if not self._check_image_exists(image_file):
                print(f'Image does not exist: {image_file}', flush=True)
                return None
            
            # Return basic information about the data without processing
            mock_data_dict.update({
                'image_path': image_file,
                'has_image': True,
                'num_conversations': len(data_dict.get('conversations', [])),
                'status': 'valid'
            })
        else:
            mock_data_dict.update({
                'has_image': False,
                'num_conversations': len(data_dict.get('conversations', [])),
                'status': 'valid'
            })
            
        return mock_data_dict
