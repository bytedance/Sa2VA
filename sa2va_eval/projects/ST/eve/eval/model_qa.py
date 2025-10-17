import argparse
import json
import os

import shortuuid
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria

from eve.conversation import conv_templates
from eve.utils import disable_torch_init
from eve.mm_utils import get_model_name_from_path


# new stopping implementation
class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


@torch.inference_mode()
def eval_model(model_path, questions_file, answers_file):
    # Model
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.float16).cuda()

    ques_file = open(os.path.expanduser(questions_file), "r")
    ans_file = open(os.path.expanduser(answers_file), "w")
    for i, line in enumerate(tqdm(ques_file)):
        idx = json.loads(line)["question_id"]
        qs = json.loads(line)["text"]
        cat = json.loads(line)["category"]
        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()
        stopping_criteria = KeywordsStoppingCriteria(
            [conv.sep], tokenizer, input_ids)
        output_ids = model.generate(
            input_ids,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            temperature=args.temperature,
            use_cache=True,
            max_new_tokens=args.max_new_tokens,
            stopping_criteria=[stopping_criteria])
        outputs = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True)[0]
        try:
            index = outputs.index(conv.sep, len(prompt))
        except ValueError:
            outputs += conv.sep
            index = outputs.index(conv.sep, len(prompt))

        outputs = outputs[len(prompt) + len(conv.roles[1]) + 2:index].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": get_model_name_from_path(model_path),
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()
    exit()
    eval_model(args.model_path, args.question_file, args.answers_file)
