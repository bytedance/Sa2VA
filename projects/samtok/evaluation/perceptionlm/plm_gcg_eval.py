import argparse
import math
import os
import torch
import tqdm
from pycocotools import mask as mask_utils
import numpy as np
import copy
import hydra

from transformers import AutoModelForImageTextToText, AutoProcessor

from PIL import Image
import re
import json

from qwen_vl_utils import process_vision_info

from projects.samtok.models import VQ_SAM2, VQ_SAM2Config, SAM2Config, DirectResize
from projects.samtok.evaluation.utils import _init_dist_pytorch, get_dist_info, collect_results_cpu


def parse_args():
    parser = argparse.ArgumentParser(description='GCG')
    parser.add_argument(
        '--model_path', 
        default='zhouyik/PLM-1B-SAMTok-co',
        help='hf model path.')
    parser.add_argument(
        '--vq_sam2_path',
        default="zhouyik/PLM-1B-SAMTok-co/mask_tokenizer_256x2.pth",
        help='vq-sam2 model path.')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--save_dir',
        default='./gcg_pred/',
        help='save path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--task_id', '--task-id', type=int, default=0)
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

# download from https://huggingface.co/datasets/Dense-World/Sa2VA-Training
IMAGE_FOLDER = './data/glamm_data/images/grandf/val_test/'


class GCGInferenceDataset:
    def __init__(self,
                 image_folder,
                 save_dir=None,
                 ):
        self.image_folder = image_folder

        self.images = os.listdir(image_folder)

        if save_dir is not None:
            # filter evaluated
            self.save_dir = save_dir
            exsits_files = os.listdir(self.save_dir)
            exsits_files = [_file[:-5] for _file in exsits_files]
            _images = []
            for i, item in enumerate(self.images):
                if item[:-4] not in exsits_files:
                    _images.append(item)
            self.images = _images

    def __len__(self):
        return len(self.images)

    def get_questions(self):
        question = "Could you please give me a brief description of the image? Please respond with interleaved \
    segmentation masks for the corresponding parts of the answer."
        return question

    def __getitem__(self, index):
        data_dict = {}
        questions = self.get_questions()
        image_file = self.images[index]
        data_dict['image_file'] = image_file

        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        data_dict['image'] = image
        data_dict['text'] = "<image>\n" + questions

        data_dict['img_id'] = image_file
        return data_dict

def extract_mt_token_ids(text):
    pattern = r"<\|mt_(\d{4})\|>"
    return [int(x) for x in re.findall(pattern, text)]


def remove_special_tokens(text):
    pattern = r"<\|mt_(start|end|\d{4})\|>"
    return re.sub(pattern, "", text)

def fix_mt_format(text):
    """
    使用正则表达式查找并修正不完整的 <|mt_...> 格式。
    这个函数会处理以下几种情况：
    1. 正确格式: <|mt_start|><|mt_0044|><|mt_0442|><|mt_end|> -> 不变
    2. 错误格式 (缺少第二个token和end): <|mt_start|><|mt_0198|> -> <|mt_start|><|mt_0198|><|mt_9999|><|mt_end|>
    3. 错误格式 (缺少第二个token但有end): <|mt_start|><|mt_0198|><|mt_end|> -> <|mt_start|><|mt_0198|><|mt_9999|><|mt_end|>
    """
    # 模式1: 匹配 <|mt_start|> + 一个token + <|mt_end|>
    # (<\|mt_start\|>) - 捕获组1: <|mt_start|>
    # (<\|mt_\d+\|>) - 捕获组2: <|mt_XXXX|>
    # (<\|mt_end\|>) - 捕获组3: <|mt_end|>
    pattern1 = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    # 替换逻辑1: 在中间插入 <|mt_-1|>
    # \1 代表捕获组1的内容, \2 代表捕获组2的内容
    replacement1 = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern1, replacement1, text)
    # 模式2: 匹配 <|mt_start|> + 一个token，且后面不是另一个mt_token或mt_end
    # (<\|mt_start\|>) - 捕获组1: <|mt_start|>
    # (<\|mt_\d+\|>) - 捕获组2: <|mt_XXXX|>
    # (?!<\|mt_) - 负向前瞻断言，确保后面不是 "<|mt_" 开头，避免匹配到正确格式的前半部分
    pattern2 = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    # 替换逻辑2: 拼接上 <|mt_-1|> 和 <|mt_end|>
    replacement2 = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern2, replacement2, text)
    return text

def fix_mt_format_comprehensive(text):
    """
    全面修正 <|mt_...> 格式的函数。
    它会处理以下几种情况：
    1. 标记太少 (1个): <|mt_start|><|mt_0198|><|mt_end|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    2. 标记太少 (1个, 无end): <|mt_start|><|mt_0198|> -> <|mt_start|><|mt_0198|><|mt_-1|><|mt_end|>
    3. 标记太多 (3个或以上): <|mt_start|><|mt_0186|><|mt_0410|><|mt_0186|><|mt_end|> -> <|mt_start|><|mt_0186|><|mt_0410|><|mt_end|>
    4. 正确格式: <|mt_start|><|mt_0044|><|mt_0442|><|mt_end|> -> 不变
    """
    # 规则 1: 处理标记太多的情况 (3个或以上)
    # 捕获前两个，匹配掉多余的，然后用前两个重构
    pattern_too_many = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_\d+\|>)(?:<\|mt_\d+\|>)+<\|mt_end\|>'
    replacement_too_many = r'\1\2\3<|mt_end|>'
    text = re.sub(pattern_too_many, replacement_too_many, text)
    # 规则 2: 处理标记太少的情况 (只有1个，且有<|mt_end|>)
    pattern_too_few_with_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(<\|mt_end\|>)'
    replacement_too_few = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_with_end, replacement_too_few, text)
    # 规则 3: 处理标记太少的情况 (只有1个，且没有<|mt_end|>)
    # 使用负向前瞻确保后面不是另一个mt_token
    pattern_too_few_no_end = r'(<\|mt_start\|>)(<\|mt_\d+\|>)(?!<\|mt_)'
    replacement_too_few_no_end = r'\1\2<|mt_9999|><|mt_end|>'
    text = re.sub(pattern_too_few_no_end, replacement_too_few_no_end, text)
    return text


def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1


    # build plm model
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=True)
    processor.image_processor.max_num_tiles = 8
    model = AutoModelForImageTextToText.from_pretrained(args.model_path).to("cuda")

    # build vq-sam2 model
    CODEBOOK_SIZE = 256
    CODEBOOK_DEPTH = 2
    sam2_config = SAM2Config(
        ckpt_path=args.sam2_path,
    )
    
    vq_sam2_config = VQ_SAM2Config(
        sam2_config=sam2_config,
        codebook_size=CODEBOOK_SIZE,
        codebook_depth=CODEBOOK_DEPTH,
        shared_codebook=False,
        latent_dim=256,
    )

    vq_sam2 = VQ_SAM2(vq_sam2_config).cuda().eval()

    state = torch.load(args.vq_sam2_path, map_location="cpu")
    vq_sam2.load_state_dict(state)

    sam2_image_processor = DirectResize(1024)


    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    dataset = GCGInferenceDataset(
        image_folder=IMAGE_FOLDER,
        save_dir=args.save_dir,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    
    rows = len(per_rank_ids)
    # chunk_size = (rows+7) // 8
    chunk_size = (rows + 0) // 1
    _start_ = args.task_id * chunk_size
    _end_ = _start_ + chunk_size
    _end_ = rows if _end_ > rows else _end_

    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        prediction = {'img_id': data_batch['img_id'], 'image_file': data_batch['image_file']}
        del data_batch['img_id'], data_batch['image_file']

        w, h = data_batch['image'].size

        image_file= prediction['image_file']
        image_file = os.path.join(IMAGE_FOLDER, image_file)
        question = data_batch['text'].replace('<image>\n', '').strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_file,
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]

        # print(messages)
        # exit(0)

        inputs = processor.apply_chat_template(
            [messages],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        try:
            inputs = inputs.to(model.device)
        except:
            prediction['prediction_masks'] = torch.zeros((0, h, w), dtype=torch.bool)
            process_and_save_output(
                args.save_dir,
                prediction['image_file'],
                '',
                prediction['prediction_masks']
            )
            results.append('')
            continue
        generate_ids = model.generate(**inputs, max_new_tokens=256)
        input_length = inputs["input_ids"].shape[1]
        generate_ids_without_inputs = generate_ids[:, input_length:]
        output_text = processor.batch_decode(generate_ids_without_inputs, skip_special_tokens=True)
        print("Assistant: ", output_text)

        quant_ids = extract_mt_token_ids(output_text[0])
        if len(quant_ids) == 0:
            print("No SEG !!!")
            prediction['prediction_masks'] = torch.zeros((0, h, w), dtype=torch.bool)
        else:
            if len(quant_ids) % CODEBOOK_DEPTH != 0:
                print("FORMAT ERROR: ", output_text)
                output_text = [fix_mt_format_comprehensive(output_text[0])]
                print("FIXED OUTPUT TEXT: ", output_text)
                quant_ids = extract_mt_token_ids(output_text[0])
            assert len(quant_ids) % CODEBOOK_DEPTH == 0
            batch_size = len(quant_ids) // CODEBOOK_DEPTH
            remap_quant_ids = []
            for bs_id in range(batch_size):
                chunk_quant_ids = quant_ids[bs_id*CODEBOOK_DEPTH:(bs_id+1)*CODEBOOK_DEPTH]
                remap_chunk_quant_ids = [quant_id - book_id*CODEBOOK_SIZE for book_id, quant_id in enumerate(chunk_quant_ids)]
                remap_chunk_quant_ids_error_handle = [quant_id if quant_id < CODEBOOK_SIZE else -1 for quant_id in remap_chunk_quant_ids]
                remap_quant_ids.append(remap_chunk_quant_ids_error_handle)

            image = Image.open(image_file).convert('RGB')
            ori_width, ori_height = image.size
            sam2_image = np.array(image)
            sam2_image = sam2_image_processor.apply_image(sam2_image)
            sam2_pixel_values = torch.from_numpy(sam2_image).permute(2, 0, 1).contiguous()
            sam2_pixel_values = sam2_pixel_values.unsqueeze(0).to(vq_sam2.dtype).to(vq_sam2.device)
            sam2_pixel_values = sam2_pixel_values.repeat(batch_size, 1, 1, 1)

            quant_ids = torch.LongTensor(remap_quant_ids).to(vq_sam2.device)

            with torch.no_grad():
                try:
                    _pred_masks = vq_sam2.forward_with_codes(sam2_pixel_values, quant_ids)
                    _pred_masks = torch.nn.functional.interpolate(_pred_masks, size=(ori_height, ori_width), mode='bilinear')
                    _pred_masks = _pred_masks > 0.5
                    _pred_masks = _pred_masks[:, 0, :, :].cpu().numpy().astype(np.uint8)
                    prediction['prediction_masks'] = torch.from_numpy(_pred_masks).to(torch.bool)
                except:
                    prediction['prediction_masks'] = torch.zeros((0, h, w), dtype=torch.bool)

        process_and_save_output(
            args.save_dir,
            prediction['image_file'],
            output_text[0],
            prediction['prediction_masks']
        )
        results.append(output_text[0])

    results = collect_results_cpu(results, len(dataset), tmpdir='./gcg_eval_tmp')


def process_and_save_output(output_dir, image_name, text_output, pred_masks):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<\|object_ref_start\|>(.*?)<\|object_ref_end\|>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    # cleaned_str = cleaned_str.replace('[SEG]', '')
    cleaned_str = remove_special_tokens(cleaned_str)

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # Convert the predicted masks into RLE format
    pred_masks_tensor = pred_masks.cpu()
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))

    # Create results dictionary
    # print(f"clean_str: {cleaned_str}")
    result_dict = {
        "image_id": image_name[:-4],
        "caption": cleaned_str,
        "phrases": phrases,
        "pred_masks": rle_masks
    }

    # print(cleaned_str)
    # print(phrases)

    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)

    return

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out

def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle

if __name__ == '__main__':
    main()
