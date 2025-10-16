import os
from typing import Literal
import pickle

import torch
import numpy as np
import copy
import json
import random
import pycocotools.mask as maskUtils

from .common import SEG_QUESTIONS, ANSWER_LIST
from .base import Sa2VABaseDataset
from .data_utils import sam2_path_patch, get_video_frames, decode_masklet, opencvimg_to_pil

class Sa2VA03RefVOS(Sa2VABaseDataset):

    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 special_tokens=None,
                 arch_type: Literal['intern_vl', 'qwen'] = 'intern_vl',
                 preprocessor=None,
                 extra_image_processor=None,
                 select_number=5,
                 sampled_frames=5,
                 dataset_type: Literal['default', 'refytvos', 'refsav']='default',
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
        
        # RefVOS-specific configurations
        self.dataset_type = dataset_type
        self.select_number = select_number
        self.sampled_frames = sampled_frames
        assert expression_file and tokenizer

        if self.dataset_type in ['default', 'refytvos']:
            assert mask_file is not None
            vid2metaid, metas, mask_dict = self.json_file_preprocess(expression_file, mask_file)
            self.video_infos = vid2metaid
            self.videos = list(self.video_infos.keys())
            self.mask_dict = mask_dict
            self.text_data = metas
        elif self.dataset_type == 'refsav':
            # prepare expression annotation files
            assert mask_file is None
            with open(expression_file, 'r') as f:
                expression_datas = json.load(f)
            self.video_infos = expression_datas
            self.videos = list(self.video_infos.keys())
            self.mask_dict = None # refsav masks are saved seperately
            self.text_data = None # text data are in the anno_dict

        self.image_folder = image_folder

    def real_len(self):
        return len(self.video_infos)

    def json_file_preprocess(self, expression_file, mask_file):
        # prepare expression annotation files
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['videos']

        metas = []
        anno_count = 0  # serve as anno_id
        vid2metaid = {}
        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]

            vid_frames = sorted(vid_express_data['frames'])
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['video'] = vid_name
                meta['exp'] = exp_dict['exp']  # str
                if self.dataset_type == 'default':
                    meta['mask_anno_id'] = exp_dict['anno_id']
                elif self.dataset_type == 'refytvos':
                    meta['mask_anno_id'] = [str(anno_count), ]
                else:
                    raise NotImplementedError

                if 'obj_id' in exp_dict.keys():
                    meta['obj_id'] = exp_dict['obj_id']
                else:
                    meta['obj_id'] = [0, ]  # Ref-Youtube-VOS only has one object per expression
                meta['anno_id'] = [str(anno_count), ]
                anno_count += 1
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id

                meta['length'] = vid_len
                metas.append(meta)
                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name] = []
                vid2metaid[vid_name].append(len(metas) - 1)
        if mask_file.endswith('.pkl'):
            with open(mask_file, 'rb') as f:
                mask_dict = pickle.load(f)
        elif mask_file.endswith('.json'):
            with open(mask_file, 'r') as f:
                mask_dict = json.load(f)
        else:
            raise NotImplementedError

        return vid2metaid, metas, mask_dict

    def decode_mask(self, video_masks, image_size):
        ret_masks = []
        for object_masks in video_masks:
            # None object
            if len(object_masks) == 0:
                if len(ret_masks) != 0:
                    _object_masks = ret_masks[0] * 0
                else:
                    _object_masks = np.zeros(
                        (self.sampled_frames, image_size[0], image_size[1]), dtype=np.uint8)
            else:
                _object_masks = []
                for i_frame in range(len(object_masks[0])):
                    _mask = np.zeros(image_size, dtype=np.uint8)
                    for i_anno in range(len(object_masks)):
                        if object_masks[i_anno][i_frame] is None:
                            continue
                        m = maskUtils.decode(object_masks[i_anno][i_frame])
                        if m.ndim == 3:
                            m = m.sum(axis=2).astype(np.uint8)
                        else:
                            m = m.astype(np.uint8)
                        _mask = _mask | m
                    _object_masks.append(_mask)
                _object_masks = np.stack(_object_masks, axis=0)
            ret_masks.append(_object_masks)
        _shape = ret_masks[0].shape
        for item in ret_masks:
            if item.shape != _shape:
                print([_ret_mask.shape for _ret_mask in ret_masks])
                return None
        ret_masks = np.stack(ret_masks, axis=0)  # (n_obj, n_frames, h, w)
        ret_masks = torch.from_numpy(ret_masks)
        ret_masks = ret_masks.flatten(0, 1)
        return ret_masks

    def prepare_data(self, index):
        """Prepare data for a given index using unified base class methods."""
        index = index % self.real_len()
        selected_video_objects = self.video_infos[self.videos[index]]
        if self.text_data is not None:
            video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

            if len(video_objects_infos) > self.select_number:
                selected_indexes = np.random.choice(len(video_objects_infos), self.select_number)
                video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
            else:
                selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=True)
                video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
            data_dict = self.dataset_map_fn(video_objects_infos, select_k=self.sampled_frames)
        else:
            assert self.dataset_type == 'refsav'
            # for refsav dataset, all the info is in the video_info rather than
            # saved to objects. We need to extract it from there.
            object_ids = list(selected_video_objects['objects'].keys())

            video_path = os.path.join(self.image_folder, selected_video_objects['video_path'])
            anno_path = os.path.join(self.image_folder, selected_video_objects['anno_path'])
            video_path, anno_path = sam2_path_patch(video_path, anno_path)

            video_frames = get_video_frames(video_path)
            video_frames = video_frames[::4]
            if not video_frames:
                return None
            # mask annotation
            with open(anno_path, 'r') as f:
                mask_data = json.load(f)
            masklets = decode_masklet(mask_data['masklet'])
            assert len(masklets) == len(video_frames), "frames should be equal to frames"
            n_objects = len(object_ids)
                # sample object
            if n_objects > self.select_number:
                selected_indexes = np.random.choice(n_objects, self.select_number)
            else:
                selected_indexes = np.random.choice(n_objects, self.select_number, replace=True)

            selected_object_ids = [object_ids[_idx] for _idx in selected_indexes]
            video_objects_infos = [selected_video_objects['objects'][_idx] for _idx in selected_object_ids]
            mask_temp_list = []
            for _mask in masklets:
                _mask_selected = []
                for _idx in selected_object_ids:
                    _mask_selected.append(_mask[:, :, int(_idx)])
                _mask_selected = np.stack(_mask_selected, axis=0)
                mask_temp_list.append(_mask_selected)
            masklets = mask_temp_list

            data_dict = self.dataset_map_fn_refsav(video_frames, masklets, video_objects_infos, select_k=self.sampled_frames)
        
        if data_dict is None:
            return None

        out_data_dict = {}
        
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('images', None) is not None:
            try:
                # Load images from paths
                if self.dataset_type in ['default', 'refytvos']:
                    images = []
                    for img_path in data_dict['images']:
                        full_img_path = os.path.join(self.image_folder, img_path)
                        image = self._read_image(full_img_path)
                        if image is None:
                            raise Exception(f"Failed to read image: {full_img_path}")
                        images.append(image)
                else:
                    assert self.dataset_type == 'refsav'
                    images = [opencvimg_to_pil(img) for img in data_dict['images']]
                    
                # Process multiple images using base class method
                image_data = self._process_multiple_images(images)
                out_data_dict.update(image_data)
                
                # Create video token string
                num_frames = len(images)
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
                
            except Exception as e:
                print(f'Error processing images: {e}', flush=True)
                return None
        else:
            # No images case
            conversations = self._process_conversations_for_encoding(data_dict['conversations'], None, is_video=True)
            token_dict = self.get_inputid_labels(conversations)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(0, 3, self.image_size, self.image_size)
            out_data_dict['masks'] = None

        out_data_dict['type'] = 'video'
        return out_data_dict
    

    def dataset_map_fn_refsav(self, video_frames, masklets, video_objects_infos, select_k=5):
        len_frames = len(masklets)
        # prepare images, random select k frames
        if len_frames > select_k + 1:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=False)
        else:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=True)
        selected_frame_indexes.sort()
        video_frames = [video_frames[_idx] for _idx in selected_frame_indexes]
        masklets = [masklets[_idx] for _idx in selected_frame_indexes]


        ret_masks = np.stack(masklets, axis=1)  # (n_obj, n_frames, h, w), align with other datasets
        ret_masks = torch.from_numpy(ret_masks)
        ret_masks = ret_masks.flatten(0, 1)

        expressions = [object_info['formated'] for object_info in video_objects_infos]

         # Convert to unified conversation format
        conversations = []
        for i, exp in enumerate(expressions):
            # the exp is a question
            question_template = random.choice(SEG_QUESTIONS)
            question = question_template.format(class_name=exp)
            if i == 0:
                # Add <image> placeholder for first question
                conversations.append({'from': 'human', 'value': '<image>\n' + question})
            else:
                conversations.append({'from': 'human', 'value': question})
            conversations.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})
        ret = {'images': video_frames, 'conversations': conversations, 'masks': ret_masks}
        return ret

    def dataset_map_fn(self, data_dict, select_k=5):
        images = []

        len_frames = len(data_dict[0]['frames'])
        for object_info in data_dict:
            assert len_frames == len(object_info['frames'])

        # prepare images, random select k frames
        if len_frames > select_k + 1:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=False)
        else:
            selected_frame_indexes = np.random.choice(len_frames, select_k, replace=True)
        selected_frame_indexes.sort()
        
        for selected_frame_index in selected_frame_indexes:
            frame_id = data_dict[0]['frames'][selected_frame_index]
            images.append(os.path.join(data_dict[0]['video'], frame_id + '.jpg'))

        # prepare text
        expressions = [object_info['exp'] for object_info in data_dict]

        # Convert to unified conversation format
        conversations = []
        for i, exp in enumerate(expressions):
            # the exp is a question
            if '?' in exp:
                question = exp
            else:
                exp = exp.replace('.', '').strip()
                question_template = random.choice(SEG_QUESTIONS)
                question = question_template.format(class_name=exp.lower())

            if i == 0:
                # Add <image> placeholder for first question
                conversations.append({'from': 'human', 'value': '<image>\n' + question})
            else:
                conversations.append({'from': 'human', 'value': question})
            conversations.append({'from': 'gpt', 'value': random.choice(ANSWER_LIST)})

        # prepare masks
        # one exp can have multiple annos
        video_masks = []
        for object_info in data_dict:
            anno_ids = object_info['mask_anno_id']
            obj_masks = []
            for anno_id in anno_ids:
                anno_id = str(anno_id)
                frames_masks = self.mask_dict[anno_id]
                frames_masks_ = []
                for frame_idx in selected_frame_indexes:
                    frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
                obj_masks.append(frames_masks_)
            video_masks.append(obj_masks)

        # read image size from the first image
        first_image_path = images[0]
        first_image_path = os.path.join(self.image_folder, first_image_path)
        first_image = self._read_image(first_image_path)
        if first_image is None:
            return None
        
        # switch height and width (PIL system (WH vs HW system)
        _image_size = first_image.size
        image_size = (_image_size[1], _image_size[0])
        masks = self.decode_mask(video_masks, image_size=image_size)
        if masks is None:
            return None

        ret = {'images': images, 'conversations': conversations, 'masks': masks}
        return ret

    @property
    def modality_length(self):
        if self.name == 'Ref-SAV':
            return [self._get_modality_length_default(20000) for _ in range(self.real_len())]
        else:
            return [self._get_modality_length_default(10000) for _ in range(self.real_len())]

    def mock_prepare_data(self, index):
        """
        Mock version of prepare_data that only checks image existence.
        Useful for testing and validation without loading full data.
        
        Returns:
            dict with status information or None if images don't exist
        """
        if self.dataset_type in ['refsav']:
            mock_data_dict = {}
            video_info = self.video_infos[self.videos[index]]
            video_path = os.path.join(self.image_folder, video_info['video_path'])
            anno_path = os.path.join(self.image_folder, video_info['anno_path'])
            video_path, anno_path = sam2_path_patch(video_path, anno_path)

            if not os.path.exists(video_path):
                print(f'Video path does not exist: {video_path}', flush=True)
                return None
            if not os.path.exists(anno_path):
                print(f'Annotation path does not exist: {anno_path}', flush=True)
                return None
            
            mock_data_dict.update({
                'video_name': video_info['video_path'],
                'has_images': True,
                'num_frames': 5,
                'num_objects': len(video_info['objects']),
                'status': 'valid',
                'type': 'video'
            })
            return mock_data_dict

        index = index % self.real_len()
        selected_video_objects = self.video_infos[self.videos[index]]
        video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

        if len(video_objects_infos) > self.select_number:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]
        else:
            selected_indexes = np.random.choice(len(video_objects_infos), self.select_number, replace=True)
            video_objects_infos = [video_objects_infos[_idx] for _idx in selected_indexes]

        mock_data_dict = {}
        
        # Check image existence
        len_frames = len(video_objects_infos[0]['frames'])
        if len_frames > self.sampled_frames + 1:
            selected_frame_indexes = np.random.choice(len_frames, self.sampled_frames, replace=False)
        else:
            selected_frame_indexes = np.random.choice(len_frames, self.sampled_frames, replace=True)
        selected_frame_indexes.sort()
        
        # Check if images exist
        for selected_frame_index in selected_frame_indexes:
            frame_id = video_objects_infos[0]['frames'][selected_frame_index]
            image_path = os.path.join(video_objects_infos[0]['video'], frame_id + '.jpg')
            full_image_path = os.path.join(self.image_folder, image_path)
            if not self._check_image_exists(full_image_path):
                print(f'Image does not exist: {full_image_path}', flush=True)
                return None
        
        mock_data_dict.update({
            'video_name': video_objects_infos[0]['video'],
            'has_images': True,
            'num_frames': len(selected_frame_indexes),
            'num_objects': len(video_objects_infos),
            'num_conversations': len(video_objects_infos) * 2,  # Each object creates 2 conversation turns
            'status': 'valid',
            'type': 'video'
        })
            
        return mock_data_dict
