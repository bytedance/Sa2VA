import os
from typing import Literal
import torch
from PIL import Image
import numpy as np
import copy
import json
from decord import VideoReader, cpu

from .base import Sa2VABaseDataset


def _get_rawvideo_dec(video_path, select_frames=5):
    """Extract frames from video using decord."""
    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    elif os.path.exists(video_path.replace('mkv', 'mp4')):
        vreader = VideoReader(video_path.replace('mkv', 'mp4'), ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0
    f_end = len(vreader) - 1
    num_frames = f_end - f_start + 1
    assert num_frames > 0, f'num_frames: {num_frames}, f_start: {f_start}, f_end: {f_end}, fps: {fps}, video_path: {video_path}'
    
    # T x 3 x H x W
    if num_frames <= select_frames:
        sample_pos = range(f_start, f_end + 1)
    else:
        split_point = np.linspace(0, num_frames, num=select_frames+1, dtype=int)
        sample_pos = [np.random.randint(split_point[i], split_point[i+1]) for i in range(select_frames)]
    
    patch_images = [Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()]
    return patch_images


class Sa2VA04VideoQA(Sa2VABaseDataset):
    """
    Sa2VA Video Question Answering Dataset.
    
    Adapted from VideoChatUniViDataset for Sa2VA architecture.
    Supports multi-turn video conversations with frame sampling.
    """

    def __init__(self,
                 image_folder,
                 json_file,
                 prompt_template=None,
                 tokenizer=None,
                 sampled_frames=10,
                 max_length=2048,
                 special_tokens=None,
                 arch_type: Literal['intern_vl', 'qwen', 'llava'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 **kwargs):
        
        # Initialize base class
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
        
        # VideoQA-specific configurations
        self.sampled_frames = sampled_frames
        
        # Load and preprocess JSON data
        assert json_file and tokenizer
        json_datas = self.json_file_preprocess(json_file)
        self.text_data = json_datas

        self.image_folder = image_folder
    
    def real_len(self):
        return len(self.text_data)

    def json_file_preprocess(self, json_file):
        """Prepare video QA annotation files."""
        with open(json_file, 'r') as f:
            json_datas = json.load(f)
        return json_datas

    def prepare_data(self, index):
        """Prepare data for a given index using unified base class methods."""
        selected_data_dict = copy.deepcopy(self.text_data[index])
        data_dict = self.dataset_map_fn(selected_data_dict, select_k=self.sampled_frames)
        
        if data_dict is None:
            return None

        out_data_dict = {}

        if data_dict.get('images', None) is not None:
            # Process multiple images using base class method
            image_data = self._process_multiple_images(data_dict['images'])
            out_data_dict.update(image_data)
            
            # Create video token string
            num_frames = len(data_dict['images'])
            image_token_str = self._create_token_string(image_data['num_image_tokens'], num_frames)
            
            # Process conversations using unified method
            conversations = self._process_conversations_for_encoding(
                data_dict['conversations'], image_token_str, is_video=True
            )
            
            # Handle token expansion for qwen if needed
            if self.arch_type == 'qwen' and 'num_frame_tokens' in image_data:
                conversations = self._expand_video_tokens(
                    conversations, image_data['num_frame_tokens'], image_data['num_image_tokens']
                )
            
            # Get input/labels using base class method
            token_dict = self.get_inputid_labels(conversations)
            out_data_dict.update(token_dict)
        else:
            # No images case
            conversations = self._process_conversations_for_encoding(data_dict['conversations'], None, is_video=True)
            token_dict = self.get_inputid_labels(conversations)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(0, 3, self.image_size, self.image_size)
        
        out_data_dict['type'] = 'video'
        return out_data_dict

    def dataset_map_fn(self, data_dict, select_k=5):
        """Map dataset function to process video and conversation data."""
        assert 'video' in data_dict
        
        # Load video frames
        video_file = data_dict['video']
        video_file = os.path.join(self.image_folder, video_file)
        images = _get_rawvideo_dec(video_file, select_frames=select_k)

        # Convert conversations to unified format
        conversations = []
        questions = []
        answers = []

        for conv in data_dict['conversations']:
            if conv['from'] == 'human':
                questions.append(conv['value'].replace('<video>', '').strip())
            else:
                answers.append(conv['value'])
        
        assert len(questions) == len(answers)

        for i, (question, answer) in enumerate(zip(questions, answers)):
            if i == 0:
                # Add <image> placeholder for first question
                conversations.append({'from': 'human', 'value': '<image>\n' + question})
            else:
                conversations.append({'from': 'human', 'value': question})
            conversations.append({'from': 'gpt', 'value': answer})

        ret = {'images': images, 'conversations': conversations}
        return ret

    def mock_prepare_data(self, index):
        """
        Mock version of prepare_data that only checks video existence.
        Useful for testing and validation without loading full data.
        
        Returns:
            dict with status information or None if video doesn't exist
        """
        selected_data_dict = copy.deepcopy(self.text_data[index])
        
        mock_data_dict = {}
        
        if 'video' in selected_data_dict:
            video_file = selected_data_dict['video']
            video_file = os.path.join(self.image_folder, video_file)
            
            # Check if video file exists (try both original and mp4)
            video_exists = os.path.exists(video_file) or os.path.exists(video_file.replace('mkv', 'mp4'))
            
            if not video_exists:
                print(f'Video does not exist: {video_file}', flush=True)
                return None
            
            mock_data_dict.update({
                'video_path': video_file,
                'has_video': True,
                'num_conversations': len(selected_data_dict.get('conversations', [])),
                'status': 'valid',
                'type': 'video'
            })
        else:
            return None
            
        return mock_data_dict

    @property
    def modality_length(self):
        return [self._get_modality_length_default(10000) for _ in range(len(self))]
