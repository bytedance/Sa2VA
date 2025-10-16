import os
import json

import mmengine

from PIL import Image
import copy

from mmengine.dist import master_only

from .base_eval_dataset import BaseEvalDataset

SEG_PROMPT = "<image>\nPlease segment {}."
SEG_PROMPT_QUESTION = "<image>\n {} Please respond with a segmentation mask."


class RefVOSDataset(BaseEvalDataset):
    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file,
    ):
        super().__init__()
        vid2metaid, metas, mask_dict = self.json_file_preprocess(expression_file, mask_file)
        self.vid2metaid = vid2metaid
        self.videos = list(self.vid2metaid.keys())
        self.mask_dict = mask_dict
        self.text_data = metas

        self.image_folder = image_folder

    def __len__(self):
        return len(self.text_data)

    def real_len(self):
        return len(self.text_data)

    def json_file_preprocess(self, expression_file, mask_file):
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['videos']
        metas = []
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
                meta['exp'] = exp_dict['exp']
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['length'] = vid_len
                metas.append(meta)
                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name] = []
                vid2metaid[vid_name].append(len(metas) - 1)

        if mask_file is not None:
            mask_dict = mmengine.load(mask_file)
        else:
            mask_dict = None
        return vid2metaid, metas, mask_dict

    def __getitem__(self, index):
        video_obj_info = copy.deepcopy(self.text_data[index])
        exp = video_obj_info['exp']

        data_dict = {}

        video_id = video_obj_info['video']
        frames_files = video_obj_info['frames']
        frames_files = [
            os.path.join(self.image_folder,video_id, frame_file + ".jpg") for frame_file in frames_files
        ]
        
        images = []
        ori_width, ori_height = None, None
        for frame_idx, frame_path in enumerate(frames_files):
            frame_image = Image.open(frame_path).convert('RGB')
            if ori_height is None:
                ori_width, ori_height = frame_image.size
            else:
                assert ori_width == frame_image.size[0]
                assert ori_height == frame_image.size[1]
            images.append(frame_image)

        data_dict['type'] = 'video'
        data_dict['index'] = index
        data_dict['video_id'] = video_id
        data_dict['images'] = images
        data_dict['exp_id'] = video_obj_info['exp_id']

        data_dict['frames'] = video_obj_info['frames']
        data_dict['text_prompt'] = SEG_PROMPT.format(exp) if '?' not in exp else SEG_PROMPT_QUESTION.format(exp)
        data_dict['image_folder'] = self.image_folder

        data_dict['length'] = video_obj_info['length']
        data_dict['ori_height'] = ori_height
        data_dict['ori_width'] = ori_width

        return data_dict
