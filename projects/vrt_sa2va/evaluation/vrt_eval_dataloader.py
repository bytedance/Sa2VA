"""
VER (Visual Evidence Reasoning) Dataset Loader
This module provides functionality to load and parse VER benchmark data from TFRecord files
and human-labeled evaluation results.
"""

import os
import json
import re
from typing import Dict, List, Tuple, Optional
from PIL import Image
from io import BytesIO
import numpy as np
from pycocotools import mask as mask_utils
from tfd_utils.random_access import TFRecordRandomAccess


class VERHumanLabelParser:
    """Parser for human-labeled VER evaluation results"""
    
    def __init__(self, label_file_paths: List[str], cls_mapping: Optional[Dict[str, Dict]] = None):
        """
        Initialize the parser with the human label files
        
        Args:
            label_file_paths: A list of paths to the human-labeled text files
        """
        self.label_file_paths = label_file_paths
        self.cls_mapping = cls_mapping
        
        
        # prepare
        self.records = self._parse_label_files()

    
    def _parse_label_files(self) -> Dict[str, Dict]:
        """Parse the human-labeled files and extract records"""
        all_records = {}
        for file_path in self.label_file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except FileNotFoundError:
                print(f"Warning: Label file not found: {file_path}")
                continue
        
            # Split by record boundaries
            record_sections = content.split('RECORD')[1:]  # Skip the header
            
            for section in record_sections:
                try:
                    record_data = self._parse_single_record(section, file_path)
                    if record_data:
                        all_records[record_data['key']] = record_data
                except Exception as e:
                    print(f"Error parsing record in {file_path}: {e}")
                    continue
                
        return all_records
    
    def _process_human_field(self, record_data: Dict, field_name: str, field_value: str):
        """Process a single human-labeled field"""
        if field_name in ['r_objs', 'a_objs']:
            # Parse object references like "<obj12>, <obj13>"
            objects = re.findall(r'<obj(\d+)>', field_value)
            record_data[f'human_labeled_{field_name}'] = [int(obj_id) for obj_id in objects]
        elif field_name == 'confidence':
            record_data['human_confidence'] = int(field_value) if field_value.isdigit() else 0
        elif field_name == 'keep':
            record_data['human_keep'] = field_value.upper() == 'Y'
        else:
            record_data[f'human_{field_name}'] = field_value
    
    def _parse_single_record(self, record_text: str, file_path: str) -> Optional[Dict]:
        """Parse a single record from the text"""
        lines = record_text.strip().split('\n')
        if not lines:
            return None
        
        # Extract record ID and key
        header_line = lines[0]
        key_match = re.search(r'KEY: (.*?)\)', header_line)
        if not key_match:
            return None
        key = key_match.group(1)
        
        record_data = {'key': key, 'source_file': os.path.basename(file_path)}

        if self.cls_mapping:
            record_data['class_ids'] = self.cls_mapping.get(key, {}).get('gemini_response', '').split(' ')
        else:
            record_data['class_ids'] = []
        
        # Parse different sections
        current_section = None
        section_content = []
        current_field = None  # Track current F_ field being parsed
        
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith('-'):
                continue
                
            # Check for section headers
            if line in ['QUESTION:', 'REASONING:', 'REASONING OBJECTS:', 
                       'ANSWER CAPTION:', 'ANSWER OBJECTS:']:
                if current_section and section_content:
                    record_data[current_section] = '\n'.join(section_content).strip()
                current_section = line.rstrip(':').lower().replace(' ', '_')
                section_content = []
                current_field = None  # Reset field when entering a new section
            elif line.startswith('F_'):
                # Process any pending section content first
                if current_section and section_content:
                    record_data[current_section] = '\n'.join(section_content).strip()
                    current_section = None
                    section_content = []
                
                # Parse labeled fields - handle both single line and multi-line formats
                if ':' in line:
                    field_parts = line.split(':', 1)
                    field_name = field_parts[0].replace('F_', '').lower()
                    field_value_part = field_parts[1].strip()
                    
                    if field_value_part.startswith('[') and field_value_part.endswith(']'):
                        # Single line format like F_CONFIDENCE:[5]
                        field_value = field_value_part[1:-1]  # Remove brackets
                        self._process_human_field(record_data, field_name, field_value)
                    else:
                        # This might be start of multi-line, set current field and continue
                        current_field = field_name
            elif line.startswith('[') and line.endswith(']') and current_field:
                # This is the value line for a multi-line F_ field
                field_value = line[1:-1]  # Remove brackets
                self._process_human_field(record_data, current_field, field_value)
                current_field = None
            elif current_section:
                section_content.append(line)
            else:
                section_content.append(line)
        
        # Add the last section
        if current_section and section_content:
            record_data[current_section] = '\n'.join(section_content).strip()
        
        # Parse object references from reasoning and answer sections
        if 'reasoning' in record_data:
            reasoning_objects = re.findall(r'<obj(\d+)>', record_data['reasoning'])
            record_data['reasoning_object_ids'] = [int(obj_id) for obj_id in reasoning_objects]
        
        if 'answer_caption' in record_data:
            answer_objects = re.findall(r'<obj(\d+)>', record_data['answer_caption'])
            record_data['answer_object_ids'] = [int(obj_id) for obj_id in answer_objects]
        
        return record_data
    
    def get_record(self, key: str) -> Optional[Dict]:
        """Get a specific record by key"""
        return self.records.get(key)
    
    def get_all_keys(self) -> List[str]:
        """Get all available keys"""
        return list(self.records.keys())
    
    def get_records_with_keep_flag(self) -> Dict[str, Dict]:
        """Get only records that should be kept (F_KEEP: [Y])"""
        return {k: v for k, v in self.records.items() 
                if v.get('human_keep', False)}


class VERTFRecordLoader:
    """Loader for VER TFRecord data"""
    
    def __init__(self, tfrecord_path: str):
        """
        Initialize the TFRecord loader
        
        Args:
            tfrecord_path: Path to the TFRecord file
        """
        self.tfrecord_path = tfrecord_path
        self.reader = TFRecordRandomAccess(tfrecord_path)
        self.keys = self.reader.get_keys()
    
    def get_data(self, key: str) -> Optional[Dict]:
        """
        Get data for a specific key
        
        Args:
            key: The key to retrieve data for
            
        Returns:
            Dictionary containing image, annotation data, and parsed VER data
        """
        if key not in self.keys:
            return None
        
        try:
            tf_feat = self.reader[key].features.feature
            
            # Parse image
            img_bytes = tf_feat['image'].bytes_list.value[0]
            image_pil = Image.open(BytesIO(img_bytes)).convert("RGB")
            
            # Parse JSON data
            json_data = json.loads(tf_feat['json'].bytes_list.value[0].decode('utf-8'))
            
            # Extract VER data (use first candidate if multiple exist)
            candidates = json_data.get('generated_ver_data', {}).get('candidates', [])
            if not candidates:
                return None
            
            ver_data = candidates[0]  # Use first candidate
            
            # Extract original object annotations
            ori_obj = json_data.get('original_annotation', {}).get('objects_anns', {})
            
            # Parse object masks and create mapping
            objects_info = {}
            for obj_id, obj_info in ori_obj.items():
                mask = mask_utils.decode(obj_info['segmentation'])
                # Extract numeric ID from object identifier (e.g., 'obj12' -> 12, '<obj12>' -> 12)
                numeric_id = re.search(r'(\d+)', obj_id)
                if numeric_id:
                    objects_info[int(numeric_id.group(1))] = {
                        'mask': mask,
                        # 'segmentation': obj_info['segmentation'],
                        'original_id': obj_id,
                        'caption': obj_info.get('caption', '')
                    }
            
            return {
                'key': key,
                'image': image_pil,
                'question': ver_data.get('question', ''),
                'reasoning': ver_data.get('reasoning', ''),
                'answer_caption': ver_data.get('answer_caption', ''),
                'answer_objects': ver_data.get('answer_objects', ''),
                'objects_info': objects_info,
                # 'raw_json': json_data
            }
            
        except Exception as e:
            print(f"Error loading data for key {key}: {e}")
            return None
    
    def get_all_keys(self) -> List[str]:
        """Get all available keys"""
        return self.keys


class VRTEvalDataset:
    """Complete VRT dataset combining TFRecord data and human labels"""
    
    def __init__(self, tfrecord_paths: List[str], human_label_paths: List[str], cls_mapping: Optional[str] = None):
        """
        Initialize the VRT dataset
        
        Args:
            tfrecord_paths: List of paths to TFRecord files
            human_label_paths: List of paths to human-labeled evaluation files
        """
        self.tfrecord_loaders = {
            i: VERTFRecordLoader(path) for i, path in enumerate(tfrecord_paths)
        }

        if cls_mapping is not None:
            _cls_mapping: List[Dict] = json.load(open(cls_mapping, 'r'))
            cls_mapping = {
                item['key']: item for item in _cls_mapping
            }

        else:
            cls_mapping = None
        self.cls_mapping = cls_mapping
        self.human_parser = VERHumanLabelParser(human_label_paths, cls_mapping=cls_mapping)
        
        # Create a mapping from key to tfrecord index
        self.key_to_loader = {}
        for loader_idx, loader in self.tfrecord_loaders.items():
            for key in loader.get_all_keys():
                self.key_to_loader[key] = loader_idx
    
    def get_sample(self, key: str) -> Optional[Dict]:
        """
        Get a complete sample with both TFRecord data and human labels
        
        Args:
            key: The sample key
            
        Returns:
            Dictionary containing all sample data
        """
        # Get TFRecord data
        if key not in self.key_to_loader:
            return None
        
        loader_idx = self.key_to_loader[key]
        tfrecord_data = self.tfrecord_loaders[loader_idx].get_data(key)
        
        if not tfrecord_data:
            return None
        
        # Get human labels
        human_data = self.human_parser.get_record(key)
        
        # Combine data
        sample = tfrecord_data.copy()
        if human_data:
            sample.update(human_data)
        
        return sample
    
    def get_evaluation_samples(self) -> List[Dict]:
        """Get all samples that should be used for evaluation (have human labels and keep flag)"""
        evaluation_samples = []
        valid_records = self.human_parser.get_records_with_keep_flag()
        
        for key in valid_records.keys():
            sample = self.get_sample(key)
            if sample:
                evaluation_samples.append(sample)
        

        # verify whether all samples have class_ids
        if self.cls_mapping is not None:
            for sample in evaluation_samples:
                assert 'class_ids' in sample and len(sample['class_ids']) > 0, \
                    f"Sample with key {sample['key']} is missing class_ids."
        return evaluation_samples
    
    def get_all_keys(self) -> List[str]:
        """Get all available keys across all TFRecord files"""
        return list(self.key_to_loader.keys())


if __name__ == "__main__":
    import glob
    tfrecord_paths = glob.glob('data/VER_BENCH/VER-Labeling/data/0801_data_label/sa2va_samples_*.tfrecord')
    human_label_paths = glob.glob('data/VER_BENCH/VER-Labeling/data/0801_data_label_result/sa2va_samples_*.txt')


    dataset = VRTEvalDataset(
        tfrecord_paths, human_label_paths, 
        cls_mapping='data/VER_BENCH/VER-Labeling/data/ver_labeling_1106.json',
    )

    dataset.get_evaluation_samples()