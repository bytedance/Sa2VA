import argparse
import json
from io import BytesIO

import requests
import torch
from PIL import Image
from tqdm import tqdm

from eve.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                               DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from eve.conversation import SeparatorStyle, conv_templates
from eve.mm_utils import (KeywordsStoppingCriteria,
                              get_model_name_from_path, tokenizer_image_token)
from eve.model.builder import load_pretrained_model
from eve.utils import disable_torch_init, process_images


def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_type)

    with open(args.questions_file) as f:
        llvqa_data = json.load(f)

    for i, llddata in enumerate(tqdm(llvqa_data)):
        filename = llddata["img_path"]
        if args.lang == "en":
            message = llddata["question"] + \
                "\nChoose between one of the options as follows:\n"
        elif args.lang == "zh":
            message = llddata["question"] + "\在下列选项中选择一个:\n"
        else:
            raise NotImplementedError(
                "Q-Bench does not support languages other than English (en) and Chinese (zh) yet. Contact us (https://github.com/VQAssessment/Q-Bench/) to convert  Q-Bench into more languages.")
        for choice, ans in zip(["A.", "B.", "C.", "D."], llddata["candidates"]):
            message += f"{choice} {ans}\n"
        qs = message
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = load_image(args.image_folder + filename)
        image_tensor = process_images([image], image_processor, None)[0]
        image_tensor = image_tensor.unsqueeze(0).half().cuda()

        input_ids = tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=args.do_sample,
                top_p=args.top_p,
                top_k=args.top_k,
                num_beams=args.num_beams,
                temperature=args.temperature,
                use_cache=True,
                max_new_tokens=args.max_new_tokens,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (
            input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 1:
            print(
                f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(
            output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        llddata["response"] = outputs
        with open(args.answers_file, "a") as wf:
            json.dump(llddata, wf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--questions-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)
