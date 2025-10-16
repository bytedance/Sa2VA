import copy
from PIL import Image
import cv2
import numpy as np
import torch
from typing import Dict, List, Sequence
from torch.nn.utils.rnn import pad_sequence
from xtuner.dataset.utils import get_bos_eos_token_ids
from xtuner.utils import IGNORE_INDEX, DEFAULT_PAD_TOKEN_INDEX
from xtuner.registry import BUILDER
from mmengine.logging import print_log
import pycocotools.mask as maskUtils
from torch.utils.data import ConcatDataset as TorchConcatDataset


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height,
                              image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image,
                       min_num=1,
                       max_num=6,
                       image_size=448,
                       use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = {(i, j)
                     for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if i * j <= max_num and i * j >= min_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio,
                                                    target_ratios, orig_width,
                                                    orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = ((i % (target_width // image_size)) * image_size,
               (i // (target_width // image_size)) * image_size,
               ((i % (target_width // image_size)) + 1) * image_size,
               ((i // (target_width // image_size)) + 1) * image_size)
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def tokenize_conversation(
        example,
        tokenizer,
        max_length,
):
    """We only support the following three scenarios:

    1. Incremental pretraining dataset.
        example['conversation'] = [
                {
                    'input': '',
                    'output': '### Human: Can you write xxx'
                }
            ]

    2. Single-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                }
            ]

    3. Multi-turn conversation dataset.
        example['conversation'] = [
                {
                    'input': 'Give three tips for staying healthy.',
                    'output': '1.Eat a balanced diet xxx'
                },
                {
                    'input': 'Please expand on the second point.',
                    'output': 'Here is an expanded explanation of the xxx'
                }
            ]
    """
    bos_token_id, eos_token_id = get_bos_eos_token_ids(tokenizer)

    input_ids, labels = [], []
    next_needs_bos_token = True
    for single_turn_conversation in example['conversation']:
        input = single_turn_conversation['input']
        input_encode = tokenizer.encode(input, add_special_tokens=False)
        if next_needs_bos_token:
            input_ids += bos_token_id
            labels += [IGNORE_INDEX] * len(bos_token_id)
        input_ids += input_encode
        labels += [IGNORE_INDEX] * len(input_encode)
        # Add output
        output_with_loss = single_turn_conversation.get(
            'output_with_loss', True)
        output = single_turn_conversation['output']
        output_encode = tokenizer.encode(output, add_special_tokens=False)
        input_ids += output_encode
        if output_with_loss:
            labels += copy.deepcopy(output_encode)
        else:
            labels += [IGNORE_INDEX] * len(output_encode)
        # Add EOS_TOKEN (with loss)
        if single_turn_conversation.get('need_eos_token', True):
            next_needs_bos_token = True
            input_ids += eos_token_id
            if output_with_loss:
                labels += copy.deepcopy(eos_token_id)
            else:
                labels += [IGNORE_INDEX] * len(eos_token_id)
        else:
            next_needs_bos_token = False
        # Add SEP (without loss)
        sep = single_turn_conversation.get('sep', '')
        if sep != '':
            sep_encode = tokenizer.encode(sep, add_special_tokens=False)
            input_ids += sep_encode
            labels += [IGNORE_INDEX] * len(sep_encode)
            

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    return {'input_ids': input_ids, 'labels': labels}



# Copyright (c) OpenMMLab. All rights reserved.
def template_map_fn(example, template):
    conversation = example.get("conversation", [])
    for i, single_turn_conversation in enumerate(conversation):
        input = single_turn_conversation.get("input", "")
        if input is None:
            input = ""
        input_text = template.INSTRUCTION.format(input=input, round=i + 1)
        system = single_turn_conversation.get("system", "")
        if system != "" and system is not None:
            system = template.SYSTEM.format(system=system)
            input_text = system + input_text
        single_turn_conversation["input"] = input_text

        if template.get("SUFFIX", None):
            output_text = single_turn_conversation.get("output", "")
            output_text += template.SUFFIX
            single_turn_conversation["output"] = output_text

        # SUFFIX_AS_EOS is False ==> need_eos_token is True
        single_turn_conversation["need_eos_token"] = not template.get(
            "SUFFIX_AS_EOS", False
        )
        single_turn_conversation["sep"] = template.get("SEP", "")

    return {"conversation": conversation}


def sa2va_collect_fn(
        instances: Sequence[Dict],
        pad_index: int = DEFAULT_PAD_TOKEN_INDEX,
        return_hf_format: bool = False,
        use_varlen_attn: bool = False
):
    assert not return_hf_format, "return_hf_format is not supported yet."
    assert not use_varlen_attn, "use_varlen_attn is not supported yet."

    input_ids, labels = [], []

    has_image = any(inst.get('pixel_values') is not None for inst in instances)
    has_pe = any(inst.get('image_grid_thw', None) is not None for inst in instances)
    has_grounding_image = any(inst.get('g_pixel_values') is not None for inst in instances)
    has_mask = any(inst.get('masks') is not None for inst in instances)


    has_vp = any(inst.get('vp_overall_mask') is not None for inst in instances)
    has_prompt_mask = any(inst.get('prompt_masks') is not None for inst in instances)
    assert has_vp and has_prompt_mask or not has_vp and not has_prompt_mask, \
    f"Inconsistent presence of visual prompts and prompt masks {has_vp} {has_prompt_mask}"

    pixel_values = []
    frames_per_batch = []
    
    image_grid_thw = []
    grounding_pixel_values = []
    object_masks = []
    vp_overall_mask = []
    prompt_masks = []
    for example in instances:
        input_ids.append(torch.LongTensor(example['input_ids']))
        labels.append(torch.LongTensor(example['labels']))

        if has_image:
            pixel_values.append(example['pixel_values'])
            if has_pe:
                image_grid_thw.append(example['image_grid_thw'])
            if has_vp:
                if 'vp_overall_mask' in example.keys() and example['vp_overall_mask'] is not None:
                    vp_overall_mask.append(example['vp_overall_mask'])
                else:
                    vp_overall_mask.append(torch.Tensor([False] * len(example['pixel_values'])))
        
        if has_grounding_image and 'g_pixel_values' in example.keys():
            if isinstance(example['g_pixel_values'], list):
                grounding_pixel_values += example['g_pixel_values']
                frames_per_batch.append(len(example['g_pixel_values']))
            else:
                grounding_pixel_values.append(example['g_pixel_values'])
                frames_per_batch.append(1)

        if has_mask:
            if 'masks' in example.keys() and example['masks'] is not None:
                if isinstance(example['masks'], list):
                    if isinstance(example['masks'][0], np.ndarray):
                        _masks = np.stack(example['masks'], axis=0)
                        _masks = torch.from_numpy(_masks)
                        object_masks.append(_masks)
                    else:
                        object_masks.append(torch.stack(example['masks'], dim=0))
                else:
                    object_masks.append(example['masks'])

        if has_prompt_mask:
            if 'prompt_masks' in example.keys():
                prompt_masks.append(example['prompt_masks'])

    ori_length = [len(ids) for ids in input_ids]
    if len(instances) > 1:
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=pad_index)
        labels = pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
    else:
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)

    # Some tokenizers have the same eos token and pad token, so input_ids
    # cannot be masked directly based on the pad token id.
    attention_mask = torch.zeros_like(input_ids).bool()
    for i, length in enumerate(ori_length):
        attention_mask[i, :length] = True

    bs, seq_len = input_ids.shape
    position_ids = torch.arange(seq_len).unsqueeze(0).long().repeat(bs, 1)

    data_dict = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'labels': labels
    }

    if has_image:
        data_dict['frames_per_batch'] = frames_per_batch
        data_dict['pixel_values'] = pixel_values
        for pixel_values_per_sample in pixel_values:
            assert isinstance(pixel_values_per_sample, torch.Tensor)
            # dim for internvl : [num_frames, 3, H, W]
            # dim for qwenvl : [L, C] C is 1176, L is length
            # assert pixel_values_per_sample.dim() == 4, "pixel_values must be a 4D tensor"
        
        if has_pe:
            data_dict['image_grid_thw'] = image_grid_thw

    if has_vp:
        data_dict['vp_overall_mask'] = torch.cat(vp_overall_mask, dim=0)

    if has_prompt_mask:
        data_dict['prompt_masks'] = prompt_masks

    if has_grounding_image:
        data_dict['g_pixel_values'] = grounding_pixel_values

    if has_mask:
        data_dict['masks'] = object_masks

    return {'data': data_dict, 'data_samples': None}


def sam2_path_patch(video_path, anno_path):
    # bugfix for video path json - remove duplicate directory structure
    # Transform './data/video_datas/sam_v_full/sav_000/sav_train/sav_000/sav_000002.mp4'
    # into './data/video_datas/sam_v_full/sav_train/sav_000/sav_000002.mp4'
    if 'sav_train' in video_path:
        path_parts = video_path.split('/')
        # Find indices of sav_train and the duplicate sav_xxx directory
        sav_train_idx = None
        duplicate_idx = None
        for i, part in enumerate(path_parts):
            if part == 'sav_train':
                assert sav_train_idx is None, "Multiple 'sav_train' directories found."
                sav_train_idx = i
        if sav_train_idx is not None:
            if path_parts[sav_train_idx - 1] == path_parts[sav_train_idx + 1]:
                duplicate_idx = sav_train_idx - 1
        
        if duplicate_idx is not None:
            del path_parts[duplicate_idx]
            video_path = '/'.join(path_parts)

            anno_parts = anno_path.split('/')
            del anno_parts[duplicate_idx]
            anno_path = '/'.join(anno_parts)
    return video_path, anno_path


def get_video_frames(video_path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return []

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

def decode_masklet(masklet):
    masks = []
    for _rle in masklet:
        mask = maskUtils.decode(_rle)
        masks.append(mask)
    return masks


def opencvimg_to_pil(image: np.ndarray) -> Image.Image:
    """Convert an OpenCV image (BGR) to a PIL image (RGB)."""
    image = image[:, :, ::-1]  # Convert BGR to RGB
    return Image.fromarray(image).convert('RGB')


class ConcatDatasetSa2VA(TorchConcatDataset):

    def __init__(self, datasets:List[dict]):
        datasets_instance = []
        for cfg in datasets:
            datasets_instance.append(BUILDER.build(cfg))
        super().__init__(datasets=datasets_instance)

        print_log(
            f'Initialized ConcatDataset with {len(datasets)} datasets.'
        )
        for dataset in self.datasets:
            print_log(f'{repr(dataset.name)}')
            print_log(f'------Number of samples: {len(dataset)}')
            print_log(f'------Real Length: {dataset.real_len()}')

    def __repr__(self):
        main_str = 'Dataset as a concatenation of multiple datasets. \n'
        main_str += ',\n'.join(
            [f'{repr(dataset)}' for dataset in self.datasets])
        return main_str
