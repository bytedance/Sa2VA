import io
import math
import base64
import torch
import copy

import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from einops import rearrange

from .base import BaseModel
from ..smp import *
from transformers import GenerationConfig, DynamicCache
from projects.ST.models.models_modeling_qwen2mm_mmrope import Qwen2MMmropeForCausalLM, AutoTokenizer


def get_transformer_and_tokenizer(model_path, tokenizer_path):
    model = Qwen2MMmropeForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, use_cache=False,
                                                    attn_implementation="sdpa")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, padding="right", trust_remote_mode=True)
    tokenizer.vis_beg_tok = "<vision>"
    tokenizer.vis_patch_tok = "<vpatch>"
    tokenizer.vis_rsep_tok = "<vrow_sep>"
    tokenizer.vis_frm_tok = "<vframe_sep>"
    tokenizer.vis_end_tok = "</vision>"
    tokenizer.vis_cls_tok = "<|vis_cls|>"

    tokenizer.vis_beg_tok_id = tokenizer.convert_tokens_to_ids("<vision>")
    tokenizer.vis_patch_tok_id = tokenizer.convert_tokens_to_ids("<vpatch>")
    tokenizer.vis_rsep_tok_id = tokenizer.convert_tokens_to_ids("<vrow_sep>")
    tokenizer.vis_frm_tok_id = tokenizer.convert_tokens_to_ids("<vframe_sep>")
    tokenizer.vis_end_tok_id = tokenizer.convert_tokens_to_ids("</vision>")
    tokenizer.vis_cls_tok_id = tokenizer.convert_tokens_to_ids("<|vis_cls|>")

    return model, tokenizer

DEFAULT_PATCH_SIZE=32
MAX_RESOLUTION=1024
VISION_TOKENS = [
    "<vision>", # vision begin
    "<vpatch>", # patch
    "<vrow_sep>", # row separator
    "<vframe_sep>", # for video use case
    "</vision>", # vision end
    "<|vis_cls|>"
    # *position_tokens,
]
NON_VISION_TOKEN_ID = -1
PROMPT_TMPL = '<|im_start|>user\n{input}<|im_end|>\n'

def load_image_to_base64(image_path: str) -> str:
    # convert image to jpeg, then to data:image/jpeg;base64,
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_string}"


def load_base64_to_PILImage(base64_string: str) -> Image:
    # convert data:image/jpeg;base64, to jpeg
    base64_string = base64_string.split(",")[1]
    decoded_string = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(decoded_string)).convert('RGB')


def get_resize_output_image_size(
    image_size, patch_size, fix_res_size=None
) -> tuple:
    
    if fix_res_size is not None:
        return fix_res_size, fix_res_size
    
    l1, l2 = image_size # 540, 32
    short, long = (l2, l1) if l2 <= l1 else (l1, l2)

    # set the nearest multiple of PATCH_SIZE for `long`
    requested_new_long = min(
        [
            math.ceil(long / patch_size) * patch_size,
            MAX_RESOLUTION,
        ]
    )

    new_long, new_short = requested_new_long, int(requested_new_long * short / long)
    
    new_short = math.ceil(new_short / patch_size) * patch_size
    return (new_long, new_short) if l2 <= l1 else (new_short, new_long)


def preprocess_image(
    image_tensor: torch.Tensor,
    patch_size: int = DEFAULT_PATCH_SIZE
) -> torch.Tensor:
    # Reshape the image to get the patches
    # shape changes: (C=3, H, W) 
    # -> (C, N_H_PATCHES, W, PATCH_H)
    # -> (C, N_H_PATCHES, N_W_PATCHES, PATCH_H, PATCH_W)
    patches = image_tensor.unfold(1, patch_size, patch_size)\
        .unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous() # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
    return patches


def get_transform(height, width):
    preprocess_transform = transforms.Compose([
            transforms.Resize((height, width)),             
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                std=[0.229, 0.224, 0.225])   # standard deviation for pre-trained models on ImageNet
        ])
    return preprocess_transform


def convert_image_base64_to_patches(base64_image: str, patch_size: int, fix_res_size: int = None) -> torch.Tensor:
    img_pil = load_base64_to_PILImage(base64_image)
    # resize the image to the nearest multiple of patch_size
    width, height = img_pil.size
    new_width, new_height = get_resize_output_image_size((width, height), patch_size=patch_size, fix_res_size=fix_res_size)
    img_tensor = get_transform(new_height, new_width)(img_pil) # 3,height, width 
    img_patches = preprocess_image(img_tensor, patch_size=patch_size) # seq_length, 64*64*3
    return img_patches


def prepare_image_textual_seq(h, w, tokenizer, add_cls=True):
    seq = ""
    tok_len = 0
    
    seq += tokenizer.vis_beg_tok
    tok_len += 1
    for _ in range(h-1):
        seq += tokenizer.vis_patch_tok * w + tokenizer.vis_rsep_tok
        tok_len += (w + 1)
    seq += tokenizer.vis_patch_tok * w + tokenizer.vis_end_tok
    tok_len += (w + 1)
    if add_cls:
        seq += tokenizer.vis_cls_tok
        tok_len += 1
    
    return seq, tok_len


def prepare_image_textual_seq_norowsep(h, w, tokenizer, add_cls=True):
    seq = ""
    tok_len = 0
    
    seq += tokenizer.vis_beg_tok
    tok_len += 1
    
    seq += tokenizer.vis_patch_tok * (w * h)
    tok_len += (w * h)
    
    seq += tokenizer.vis_end_tok
    tok_len += 1
    
    if add_cls:
        seq += tokenizer.vis_cls_tok
        tok_len += 1
    
    return seq, tok_len


def create_single_prefix_mask(prefix_len, max_len):
    attn_mask = torch.zeros(max_len, max_len)
    attn_mask[:prefix_len, :prefix_len] = 1
    causal_mask = torch.tril(torch.ones(max_len, max_len))
    attn_mask = attn_mask.bool() | causal_mask.bool()
    return attn_mask


def generate_mm_pos_ids_singleit(input_ids, vpatch_id, h, w):
    input_ids_pt = torch.Tensor(input_ids).int()
    vpatch_pos = torch.argwhere(input_ids_pt == vpatch_id)
    vpatch_start_pos = vpatch_pos[0].item()
    nt = len(input_ids) - (h*w) + 1
 
    # v_pos
    t_indices = torch.arange(1)
    h_indices = torch.arange(h)
    w_indices = torch.arange(w)
    v_pos_id = torch.stack(torch.meshgrid(t_indices, h_indices, w_indices, indexing='ij'), dim=0)
    v_pos_id = rearrange(v_pos_id, "d t h w -> (t h w) d")  # [h*w, 3]
    v_pos_id += vpatch_start_pos
    position_id = torch.cat(
        [
            torch.arange(vpatch_start_pos).unsqueeze(-1).repeat(1,3),
            v_pos_id,
            torch.arange(nt-vpatch_start_pos-1).unsqueeze(-1).repeat(1,3) + v_pos_id.max() + 1,
        ],
        dim=0
    )
    assert len(input_ids) == position_id.size(0)
    position_id = rearrange(position_id, "slen d -> d slen").long()
    
    return position_id


class Qwen2mmMROPEModel(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='/mnt/bn/zilongdata-us/weixian/ckpt/qwen2mm-7B-mrope', tokenizer_path="/mnt/bn/zilongdata-us/weixian/ckpt/Qwen2.5MM-7B-ext-psz16", fix_res_size=None, **kwargs):

        model, tokenizer = get_transformer_and_tokenizer(
            model_path, tokenizer_path
        )
        self.model = model.cuda().eval()
        self.tokenizer = tokenizer
        
        self.image_processor = lambda x: convert_image_base64_to_patches(load_image_to_base64(x),  model.config.vision_patch_size, fix_res_size=fix_res_size)
        self.kwargs = kwargs
    
    def prepare_input(self, image, text_input):

        text_input = text_input.replace("<image>\n", '').replace("\n<image>", '').replace("<image> ", '').replace(" <image>", '')
        bos_token = '' if self.tokenizer.bos_token is None else self.tokenizer.bos_token
        text_input = bos_token + PROMPT_TMPL.format(input=text_input.strip())      

        if image is not None:
            tokens = []
            vision_patch_indices = []
            vision_patches = []
            
            patches = image
            n_rows, n_cols = patches.shape[:2]
            n_patches = n_rows * n_cols
            patches = patches.view(n_patches, -1)

            # ---
            image_text_seq, image_tok_len = prepare_image_textual_seq_norowsep(n_rows, n_cols, self.tokenizer, add_cls=False)
            # ---
            cur_tokens_pt = self.tokenizer(image_text_seq, add_special_tokens=False, return_tensors="pt").input_ids.squeeze(0)
            cur_patch_indices = torch.full_like(cur_tokens_pt, fill_value=NON_VISION_TOKEN_ID)
            assert (cur_tokens_pt==self.tokenizer.vis_patch_tok_id).sum() == n_patches
            assert (cur_tokens_pt>=self.tokenizer.vis_beg_tok_id).sum() == image_tok_len
            cur_patch_indices[cur_tokens_pt==self.tokenizer.vis_patch_tok_id] = torch.arange(n_patches)
            
            cur_tokens = cur_tokens_pt.cpu().numpy().tolist()
            cur_patch_indices = cur_patch_indices.cpu().numpy().tolist()
            assert len(cur_tokens) == len(cur_patch_indices)
            
            tokens.extend(cur_tokens)
            vision_patch_indices.extend(cur_patch_indices)
            vision_patches.extend(patches.numpy().astype(np.float16))
            
            # For text after images
            _tokenized_text = self.tokenizer(text_input, return_tensors="pt", add_special_tokens=False)
            cur_tokens = _tokenized_text["input_ids"].squeeze(0)
            tokens.extend(cur_tokens)
            vision_patch_indices.extend([NON_VISION_TOKEN_ID] * len(cur_tokens))

            position_ids = generate_mm_pos_ids_singleit(tokens, self.tokenizer.vis_patch_tok_id, n_rows, n_cols)  # [3, slen]
            attention_mask_4d = create_single_prefix_mask(image_tok_len, len(tokens)).unsqueeze(0) # [1, slen, slen]

            tokens = torch.Tensor(tokens).long()
            vision_patch_indices = torch.Tensor(vision_patch_indices).long()
            if len(vision_patches) > 0:
                # convert vision patches to numpy
                vision_patches = np.array(vision_patches)
                vision_patches = torch.Tensor(vision_patches).bfloat16()
            else:
                vision_patches = None

            tokens = tokens.unsqueeze(0)
            position_ids = position_ids.unsqueeze(1)
            attention_mask_4d = attention_mask_4d.unsqueeze(0)
            vision_patch_indices = vision_patch_indices.unsqueeze(0)
            attn_mask_for_gen = torch.ones_like(tokens)

            return dict(
                input_ids = tokens.to("cuda"),
                position_ids = position_ids.to("cuda"),
                attention_mask = attn_mask_for_gen.to("cuda"),
                vision_patches = vision_patches.to("cuda"),
                vision_patch_indices = vision_patch_indices.to("cuda"),
                attention_mask_4d = attention_mask_4d.to("cuda"),
                image_tokens_len = image_tok_len
            )

        # image is None
        _text_inputs = self.tokenizer(text_input, return_tensors="pt", add_special_tokens=False)
        text_input_ids = _text_inputs['input_ids']
        text_attn_mask = _text_inputs['attention_mask']
        text_position_ids = torch.arange(text_input_ids.size(-1)).unsqueeze(0).expand(3,-1).clone().long()
        return dict(
            input_ids = text_input_ids.long().to("cuda"),
            attention_mask = text_attn_mask.long().to("cuda"),
            position_ids = text_position_ids.unsqueeze(1).to("cuda"),
            vision_patches = None,
            vision_patch_indices = None,
            attention_mask_4d = None,
            image_tokens_len = None
        )


    def generate_inner(self, message, dataset=None):
        prompt, image_path = self.message_to_promptimg(message, dataset=dataset)
        image_patches = None if image_path is None else \
            self.image_processor(image_path)
        inputs = self.prepare_input(image_patches, prompt)

        past_key_values = None
        image_tok_len = inputs.pop("image_tokens_len")
        attention_mask_4d = inputs.pop("attention_mask_4d")
        if image_tok_len is not None and attention_mask_4d is not None:
            assert (attention_mask_4d[:,:,:image_tok_len,:image_tok_len]==1).all()
            assert inputs["vision_patches"] is not None
            assert inputs["vision_patch_indices"] is not None
            prefix_cache = DynamicCache()
            cache_inputs = dict(
                input_ids = inputs['input_ids'][:, :image_tok_len],
                position_ids = inputs['position_ids'][:, :, :image_tok_len],
                attention_mask = attention_mask_4d[:,:,:image_tok_len,:image_tok_len],
                vision_patches = inputs['vision_patches'],
                vision_patch_indices = inputs['vision_patch_indices'][:, :image_tok_len],
            )
            with torch.no_grad():
                prefix_cache = self.model(**cache_inputs, past_key_values=prefix_cache, use_cache=True).past_key_values
            past_key_values = copy.deepcopy(prefix_cache)
        
        generation_args = GenerationConfig(
                do_sample=False,
                top_p=None,
                temperature=0,
                num_beams=1,
                max_new_tokens=128,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,  
            )
        
        generate_ids = self.model.generate(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            generation_config=generation_args
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response
    

    def generate_ext_eval(self, args, prompt, image_path=None, generate_config=None):
        image_patches = None if image_path is None else \
            self.image_processor(image_path)
        inputs = self.prepare_input(image_patches, prompt)

        past_key_values = None
        image_tok_len = inputs.pop("image_tokens_len")
        attention_mask_4d = inputs.pop("attention_mask_4d")
        if image_tok_len is not None and attention_mask_4d is not None:
            assert (attention_mask_4d[:,:,:image_tok_len,:image_tok_len]==1).all()
            assert inputs["vision_patches"] is not None
            assert inputs["vision_patch_indices"] is not None
            prefix_cache = DynamicCache()
            cache_inputs = dict(
                input_ids = inputs['input_ids'][:, :image_tok_len],
                position_ids = inputs['position_ids'][:, :, :image_tok_len],
                attention_mask = attention_mask_4d[:,:,:image_tok_len,:image_tok_len],
                vision_patches = inputs['vision_patches'],
                vision_patch_indices = inputs['vision_patch_indices'][:, :image_tok_len],
            )
            with torch.no_grad():
                prefix_cache = self.model(**cache_inputs, past_key_values=prefix_cache, use_cache=True).past_key_values
            past_key_values = copy.deepcopy(prefix_cache)
        
        generation_args = GenerationConfig(
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,  
            ) if generate_config is None else GenerationConfig(**generate_config)
        
        generate_ids = self.model.generate(
            **inputs,
            past_key_values=past_key_values,
            use_cache=True,
            eos_token_id=self.tokenizer.eos_token_id,
            generation_config=generation_args
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.tokenizer.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return response