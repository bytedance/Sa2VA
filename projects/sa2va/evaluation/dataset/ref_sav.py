import logging
import os
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict
from mmengine import print_log
import mmengine
from PIL import Image
import numpy as np
from mmengine.dist import master_only

from xtuner.registry import BUILDER
from xtuner.dataset.huggingface import build_origin_dataset
import copy

from vlm.datasets.evaluation.base_eval_dataset import BaseEvalDataset
from projects.sa2va.datasets.encode_fn import video_lisa_encode_multi_conv_fn
import json
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

SEG_QUESTIONS = [
    "Please segment the object according to the description: {class_name}",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]


import cv2
def get_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return
    frames = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        frame_id += 1
    cap.release()
    return frames

def multi_template_fn(conversations, template_map):
    for conv in conversations:
        for i, single_turn_conversation in enumerate(conv):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = template_map.INSTRUCTION.format(input=input, round=i + 1)
            system = single_turn_conversation.get('system', '')
            if system != '' and system is not None:
                system = template_map.SYSTEM.format(system=system)
                input_text = system + input_text
            single_turn_conversation['input'] = input_text

            if template_map.get('SUFFIX', None):
                output_text = single_turn_conversation.get('output', '')
                output_text += template_map.SUFFIX
                single_turn_conversation['output'] = output_text

            # SUFFIX_AS_EOS is False ==> need_eos_token is True
            single_turn_conversation['need_eos_token'] = \
                not template_map.get('SUFFIX_AS_EOS', False)
            single_turn_conversation['sep'] = template_map.get('SEP', '')


class VideoRefSAM2EvalDataset(BaseEvalDataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_START_TOKEN = '<img>'
    IMG_END_TOKEN = '</img>'

    FAST_IMG_CONTEXT_TOKEN = '<FAST_IMG_CONTEXT>'
    FAST_IMG_START_TOKEN = '<fast_img>'
    FAST_IMG_END_TOKEN = '</fast_img>'

    METAINFO: dict = dict(name='revos')

    def __init__(self,
                 image_folder,
                 expression_file,
                 mask_file,
                 extra_image_processor=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 template_map_fn=None,
                 max_length=2048,
                 lazy=True,
                 special_tokens=None,
                 # eval settings
                 num_frames=5,
                 # eval name
                 eval_name=None,
                 use_fast=False,
                 fast_pool_size=2,
    ):
        super().__init__()
        assert lazy is True
        self.tokenizer = BUILDER.build(tokenizer)
        assert offline_processed_text_folder or (expression_file and tokenizer)
        self.lazy = lazy

        self.max_length = max_length

        self.template_map = template_map_fn['template']

        if offline_processed_text_folder and expression_file:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            vid2metaid, metas, mask_dict = self.json_file_preprocess(expression_file, mask_file)
            self.vid2metaid = vid2metaid
            self.videos = list(self.vid2metaid.keys())
            self.mask_dict = mask_dict
            self.json_datas = metas
            json_datas = metas
            self.text_data = json_datas
            # json_data = DatasetDict({'train': HFDataset.from_list(json_datas)})
            # if self.lazy:
            #     self.text_data = build_origin_dataset(json_data, 'train')
            # else:
            #     raise NotImplementedError

        self.image_folder = image_folder
        if extra_image_processor is not None:
            self.extra_image_processor = BUILDER.build(extra_image_processor)
        self.down_ratio = 1
        self.repeats = 1

        self._system = ''

        self.downsample_ratio = 0.5
        self.image_size = 448
        patch_size = 14
        self.patch_token = int((self.image_size // patch_size) ** 2 * (self.downsample_ratio ** 2))

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        ])

        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)


        self.num_frames = num_frames

        self.use_fast = use_fast
        self.fast_pool_size = fast_pool_size

        # save to json
        if eval_name is None:
            eval_name = 'results'
        self.eval_name = eval_name

    def __len__(self):
        return len(self.vid2metaid) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.vid2metaid:
            cur_len = 10000
            length_list.append(cur_len)
        return length_list

    def real_len(self):
        return len(self.vid2metaid)

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
                meta['mask_anno_id'] = exp_dict['anno_id']

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

        # process mask annotation files
        with open(mask_file, 'rb') as f:
            mask_dict = json.load(f)

        return vid2metaid, metas, mask_dict

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def dataset_map_fn(self, data_dict):
        images = []

        len_frames = len(data_dict[0]['frames'])
        for objet_info in data_dict:
            assert len_frames == len(objet_info['frames'])
        selected_frame_indexes = range(len_frames)

        for selected_frame_index in selected_frame_indexes:
            frame_id = data_dict[0]['frames'][selected_frame_index]
            images.append(os.path.join(data_dict[0]['video'], frame_id + '.jpg'))
        num_frames = len(images) if len(images) < self.num_frames else self.num_frames
        num_fast_frames = len(images)

        # prepare text
        expressions = [object_info['exp'] for object_info in data_dict]
        # Modify: To n dialogues
        text_dict = self.prepare_text(num_frames, expressions, num_image_tokens=self.patch_token,
                                      num_fast_frames=num_fast_frames)
        ret = {'images': images, 'video_masks': None, 'conversation': text_dict['conversation']}

        return ret

    def prepare_text(self, n_frames, expressions, num_image_tokens=256, num_fast_frames=0):

        if self.use_fast:
            fast_frame_token_str = f'{self.FAST_IMG_START_TOKEN}' \
                                   f'{self.FAST_IMG_CONTEXT_TOKEN * num_fast_frames * self.fast_pool_size * self.fast_pool_size}' \
                                   f'{self.FAST_IMG_END_TOKEN}' + '\n'
        else:
            fast_frame_token_str = ''

        frame_token_str = f'{self.IMG_START_TOKEN}' \
                          f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}' \
                          f'{self.IMG_END_TOKEN}'

        questions = []
        for i, exp in enumerate(expressions):
            question_template = SEG_QUESTIONS[0]
            questions.append(question_template.format(class_name=exp))

        eval_conversation_list = []
        for i, question in enumerate(questions):
            qa_list = []
            frame_tokens = frame_token_str + '\n'
            frame_tokens = frame_tokens * n_frames
            frame_tokens = frame_tokens.strip()
            qa_list.append(
                {'from': 'human', 'value': fast_frame_token_str + frame_tokens + question}
            )
            qa_list.append(
                {'from': 'gpt', 'value': ''}
            )
            assert len(qa_list) == 2

            input = ''
            conversation = []
            for msg in qa_list:
                if msg['from'] == 'human':
                    input += msg['value']
                elif msg['from'] == 'gpt':
                    if msg['value'] == '':
                        conversation.append({'input': input,})
                    else:
                        conversation.append({'input': input, 'output': msg['value']})
                    input = ''
                else:
                    raise NotImplementedError

            # add system information
            conversation[0].update({'system': self._system})
            eval_conversation_list.append(conversation)
        return {'conversation': eval_conversation_list}

    def __getitem__(self, index):
        index = index % self.real_len()
        selected_video_objects = self.vid2metaid[self.videos[index]]
        video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]

        selected_objects = video_objects_infos
        text_prompts = [copy.deepcopy(item['exp']) for item in selected_objects]

        data_dict = self.dataset_map_fn(selected_objects)
        multi_template_fn(data_dict['conversation'], self.template_map)
        result = video_lisa_encode_multi_conv_fn(data_dict, input_ids_with_output=False, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)

        assert 'images' in data_dict.keys()
        pixel_values = []
        if self.use_fast:
            fast_pixel_values = []
        extra_pixel_values = []
        if data_dict.get('images', None) is not None:
            # frames_files = data_dict['images']
            # frames_files = [os.path.join(self.image_folder, frame_file) for frame_file in frames_files]
            video_name = data_dict['images'][0].split('/')[0]
            video_folder = video_name[:len('sav_055')]
            video_path = os.path.join(self.image_folder, video_folder, video_name + '.mp4')
            frames = get_video_frames(video_path)

            frame_ids = [itm[len('sav_055000/1'):-len('.jpg')] for itm in data_dict['images']]
            frame_ids = [int(itm) for itm in frame_ids]

            ori_width, ori_height = None, None
            for frame_idx, frame_id in enumerate(frame_ids):
                frame_image = frames[frame_id]
                frame_image = frame_image[:, :, ::-1]
                frame_image = Image.fromarray(frame_image).convert('RGB')
                if ori_height is None:
                    ori_width, ori_height = frame_image.size
                else:
                    assert ori_width == frame_image.size[0]
                    assert ori_height == frame_image.size[1]

                if self.extra_image_processor is not None:
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_pixel_values)
                if self.use_fast:
                    frame_image = self.transformer(frame_image)
                    fast_pixel_values.append(frame_image)
                    if frame_idx < self.num_frames:
                        pixel_values.append(frame_image)
                else:
                    if frame_idx < self.num_frames:
                        frame_image = self.transformer(frame_image)
                        pixel_values.append(frame_image)

            pixel_values = torch.stack(pixel_values, dim=0) # (n_f, 3, h, w)
            data_dict['pixel_values'] = pixel_values
            if self.use_fast:
                fast_pixel_values = torch.stack(fast_pixel_values, dim=0)  # (n_f, 3, h, w)
                data_dict['fast_pixel_values'] = fast_pixel_values
            if self.extra_image_processor is not None:
                data_dict['g_pixel_values'] = extra_pixel_values
        else:
            data_dict['pixel_values'] = torch.zeros(0, 3, self.image_size, self.image_size)
            ori_width, ori_height = None, None

        data_dict['type'] = 'video'
        data_dict['video_id'] = index
        data_dict['text_prompts'] = text_prompts
        data_dict['image_folder'] = self.image_folder
        data_dict['ori_height'] = ori_height
        data_dict['ori_width'] = ori_width
        data_dict['id'] = index

        return data_dict

    @master_only
    def evaluate(self, results, work_dir):
        final_results = {}
        for idx, item in enumerate(results):
            _id = item['id']
            # vid_id = self.videos[idx]
            vid_id = self.videos[_id]
            selected_video_objects = self.vid2metaid[vid_id]
            video_objects_infos = [copy.deepcopy(self.text_data[idx]) for idx in selected_video_objects]
            text_prompts = [copy.deepcopy(item['exp']) for item in video_objects_infos]
            exp_ids = [copy.deepcopy(item['exp_id']) for item in video_objects_infos]
            final_results[vid_id] = {}
            assert len(text_prompts) == len(item['prediction_masks']), f"{len(text_prompts)}-----{len(item['prediction_masks'])}"
            for idt, text in enumerate(text_prompts):
                exp_id = exp_ids[idt]
                final_results[vid_id][exp_id] = {
                    'exp': text,
                    'prediction_masks': item['prediction_masks'][idt],
                }

        mmengine.dump(final_results, os.path.join(work_dir, f'{self.eval_name}.json'))
        return {"Dummy": 0}




if __name__ == '__main__':
    data_root_refsam2 = 'data/ref_sav/val/'
    video_refsam2_image_folder = 'data/ref_sav/sam_v_full/sav_train/'
    video_refsam2_expression_file = data_root_refsam2 + 'meta_expressions_valid.json'
    video_refsam2_mask_file = data_root_refsam2 + 'mask_dict.json'

    path = './pretrained/internvl/InternVL2-1B/'

    from transformers import AutoTokenizer

    tokenizer = dict(
        type=AutoTokenizer.from_pretrained,
        pretrained_model_name_or_path=path,
        trust_remote_code=True,
        padding_side='right'
    )

    from projects.sa2va.models.preprocess.image_resize import DirectResize
    extra_image_processor = dict(
        type=DirectResize,
        target_length=1024,
    )
    from xtuner.utils import PROMPT_TEMPLATE
    prompt_template = PROMPT_TEMPLATE.internlm2_chat

    from xtuner.dataset.map_fns import template_map_fn_factory

    dataset = VideoRefSAM2EvalDataset(
        image_folder=video_refsam2_image_folder,
        expression_file=video_refsam2_expression_file,
        mask_file=video_refsam2_mask_file,
        tokenizer=tokenizer,
        template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
        lazy=True,
        max_length=8196,
        special_tokens=['[SEG]'],
        extra_image_processor=extra_image_processor,
        eval_name='refsam2'
    )


    for item in dataset:
        print(len(item['text_prompts']))
        print(item['text_prompts'])