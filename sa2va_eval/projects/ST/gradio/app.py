import gradio as gr
import sys
from projects.ST.gradio.app_utils import\
    process_markdown, show_mask_pred, description, preprocess_video,\
    show_mask_pred_video, image2video_and_save

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
import argparse
import os

from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from gradio_image_prompter import ImagePrompter
from transformers import SamModel, SamProcessor
from xtuner.dataset.utils import load_image
import numpy as np

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_visual_prompts(points):
    ret = {'points': [], 'boxes': []}
    for item in points:
        if item[2] == 1.0:
            ret['points'].append([item[0], item[1]])
        elif item[2] == 2.0 or item[2] == 3.0:
            ret['boxes'].append([item[0], item[1], item[3], item[4]])
        else:
            raise NotImplementedError
    return ret

def parse_args(args):
    parser = argparse.ArgumentParser(description="ST Demo")
    parser.add_argument('config', help='config path.')
    parser.add_argument('pth_model', help='pth path.')
    return parser.parse_args(args)

def inference(image_dict, follow_up, input_str):
    image = image_dict['image']
    image = load_image(image)
    prompts = image_dict['points']
    visual_prompts = parse_visual_prompts(prompts)

    input_image = image

    # get the mask from visual prompts
    point_prompt_masks = []
    box_prompt_masks = []
    prompt_idxs = []
    replaced_dict = {}

    cur_prompt_id = 0
    max_prompt_id = 15

    for i, point_prompt in enumerate(visual_prompts['points']):
        input_points = [[point_prompt]]
        inputs = sam_processor(image, input_points=input_points, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores.cpu()
        masks = [item.cpu() for item in masks]
        scores = scores[0, 0]
        _masks = masks[0][0][scores == torch.max(scores)].cpu().numpy()[0]
        point_prompt_masks.append(_masks)
        assert cur_prompt_id < max_prompt_id
        replaced_dict[f"<point{i+1}>"] = f"<Prompt{cur_prompt_id}>"
        prompt_idxs.append(cur_prompt_id)
        cur_prompt_id += 1
    for i, box_prompt in enumerate(visual_prompts['boxes']):
        input_boxes = [[box_prompt]]
        inputs = sam_processor(image, input_boxes=input_boxes, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = sam_model(**inputs)
        masks = sam_processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores.cpu()
        masks = [item.cpu() for item in masks]
        scores = scores[0, 0]
        _masks = masks[0][0][scores == torch.max(scores)].cpu().numpy()[0]
        box_prompt_masks.append(_masks)
        assert cur_prompt_id < max_prompt_id
        replaced_dict[f"<box{i + 1}>"] = f"<Prompt{cur_prompt_id}>"
        prompt_idxs.append(cur_prompt_id)
        cur_prompt_id += 1

    follow_up = False

    if not follow_up:
        # reset
        print('Log: History responses have been removed!')
        global_infos.n_turn = 0
        global_infos.inputs = ''
        text = input_str

        image = input_image
        global_infos.image_for_show = image
        global_infos.image = image

        if image is not None:
            global_infos.input_type = "image"
        else:
            global_infos.input_type = "video"

    else:
        text = input_str
        image = global_infos.image

    past_text = global_infos.inputs

    if past_text == "" and "<image>" not in text:
        text = "<image>" + text

    for replace_key in replaced_dict.keys():
        text = text.replace(replace_key, replaced_dict[replace_key])

    input_dict = {
        'image': image,
        'text': text,
        'past_text': past_text,
        "prompt_masks": point_prompt_masks + box_prompt_masks,
        "prompt_ids": prompt_idxs,
    }

    return_dict = sa2va_model.predict_forward(**input_dict)
    global_infos.inputs = ""
    print(return_dict.keys())

    if 'prediction_masks' in return_dict.keys() and return_dict['prediction_masks'] and len(
            return_dict['prediction_masks']) != 0:
        print(len(return_dict['prediction_masks']), return_dict['prediction_masks'][0].shape)
        image_mask_show, selected_colors = show_mask_pred(global_infos.image_for_show, return_dict['prediction_masks'],)
    else:
        image_mask_show = global_infos.image_for_show
        selected_colors = []

    predict = return_dict['prediction'].strip()
    for replace_key in replaced_dict.keys():
        predict = predict.replace(replaced_dict[replace_key], replace_key)

    global_infos.n_turn += 1

    predict = process_markdown(predict, selected_colors)
    return image_mask_show, predict

def init_models(args):
    # load config
    cfg = Config.fromfile(args.config)
    # if args.cfg_options is not None:
    # cfg.merge_from_dict(args.cfg_options)

    cfg.model.pretrained_pth = None

    model = BUILDER.build(cfg.model)

    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    # del state_dict['llm.base_model.model.model.tok_embeddings.weight']
    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    model.cuda()
    model.eval()
    model.preparing_for_generation(metainfo={})
    return model

class global_infos:
    inputs = ''
    n_turn = 0
    image_width = 0
    image_height = 0

    image_for_show = None
    image = None
    video = None

    input_type = "image" # "image" or "video"

if __name__ == "__main__":
    # get parse args and set models
    args = parse_args(sys.argv[1:])

    sa2va_model = \
        init_models(args)

    sam_model = SamModel.from_pretrained("./pretrained/sam-vit-huge").cuda()
    sam_processor = SamProcessor.from_pretrained("./pretrained/sam-vit-huge")

    demo = gr.Interface(
        inference,
        inputs=[
            # gr.Image(type="pil", label="Upload Image", height=360),
            ImagePrompter(
                type='filepath', label='Input Image (Please click points or draw bboxes)', interactive=True,
                elem_id='image_upload', height=360, visible=True, render=True
            ),
            gr.Checkbox(label="Follow up Question"),
            gr.Textbox(lines=1, placeholder=None, label="Text Instruction"),],
        outputs=[
            gr.Image(type="pil", label="Output Image"),
            gr.Markdown()],
        theme=gr.themes.Soft(), allow_flagging="auto", description=description,
        title='Sa2VA'
    )

    demo.queue()
    demo.launch(share=False)