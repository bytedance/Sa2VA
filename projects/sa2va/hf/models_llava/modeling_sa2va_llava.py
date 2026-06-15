import torch
from torch import nn
from transformers import GenerationConfig, LlavaForConditionalGeneration
from transformers.modeling_utils import PreTrainedModel

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from .configuration_sa2va_chat import Sa2VAChatConfigLlava

from .sam2 import SAM2
from .templates import PROMPT_TEMPLATE

import numpy as np
from torchvision.transforms.functional import to_pil_image

import torch.nn.functional as F


class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))


class Sa2VAChatModelLlava(PreTrainedModel):
    config_class = Sa2VAChatConfigLlava
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['CLIPEncoderLayer', 'LlamaDecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    # OpenAI CLIP normalization (LLaVA-1.5 vision tower); must mirror training
    # (Sa2VADatasetMixin.CLIP_MEAN/STD).
    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(self, config: Sa2VAChatConfigLlava, model=None, use_flash_attn=True):
        super().__init__(config)
        self.extra_image_processor = DirectResize(target_length=1024, )

        self.torch_dtype = torch.bfloat16

        if model is not None:
            self.model = model
        else:
            self.model = LlavaForConditionalGeneration(config)

        # 576 for CLIP-ViT-L-336 (336/14)^2; LLaVA-1.5 uses every patch token
        self.image_size = config.vision_config.image_size
        self.patch_token = (config.vision_config.image_size
                            // config.vision_config.patch_size) ** 2
        self.IMG_CONTEXT_TOKEN = '<image>'

        # vicuna prompt template (must match training: bare 'USER: ... ASSISTANT:',
        # no system prompt) — do NOT use llava-hf's bundled chat template.
        template = config.template.replace('-', '_')
        self.template = PROMPT_TEMPLATE[template]

        self.transformer = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=self.CLIP_MEAN, std=self.CLIP_STD)
        ])

        llm_hidden_size = config.text_config.hidden_size

        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        self.init_prediction_config = False

    @property
    def lm_head(self):
        return self.model.lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    def preparing_for_generation(self, tokenizer, max_new_tokens=2048):
        # set generation configs for the model; vicuna has no extra stop words,
        # generation stops at </s> (matches training, which appends </s>).
        self.tokenizer = tokenizer
        self.seg_token_idx = tokenizer.convert_tokens_to_ids('[SEG]')
        self.gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
        )
        self.init_prediction_config = True

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            processor=None,
    ):
        # tokenizer-only path: eval scripts pass processor=None for non-qwen models
        if not self.init_prediction_config:
            assert tokenizer is not None
            self.preparing_for_generation(tokenizer=tokenizer)

        assert mask_prompts is None, "Visual prompts are not supported for the LLaVA backbone."

        if image is None and video is None and '<image>' not in past_text:
            text = text.replace('<image>', "")
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(input=text, round=1)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).to(self.device).unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': None,
                'input_ids': ids,
                'attention_mask': attention_mask,
            }
            ret_masks = []
        else:
            input_dict = {}
            if video is not None:
                pixel_values = []
                extra_pixel_values = []
                ori_image_size = video[0].size
                for frame_idx, frame_image in enumerate(video):
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_image)
                    if frame_idx < 5:
                        img = self.transformer(frame_image)
                        pixel_values.append(img)

                pixel_values = torch.stack(pixel_values, dim=0).to(self.torch_dtype)  # (n_f, 3, h, w)
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)
                num_frames = len(pixel_values)
            else:
                ori_image_size = image.size

                # prepare grounding images
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
                extra_pixel_values = [g_pixel_values]
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                pixel_values = self.transformer(image).unsqueeze(0).to(self.torch_dtype)  # (1, 3, h, w)
                num_frames = 1
            input_dict['g_pixel_values'] = g_pixel_values
            input_dict['pixel_values'] = pixel_values

            # one '<image>' literal per vision patch token; HF llava scatters the
            # vision features into exactly these positions (hard count check)
            image_token_str = self.IMG_CONTEXT_TOKEN * self.patch_token + '\n'
            image_token_str = image_token_str * num_frames
            image_token_str = image_token_str.strip()

            ret_masks = []

            if '<image>' in text:
                assert past_text is None or len(past_text) == 0
                text = text.replace('<image>', image_token_str)
            else:
                text = image_token_str + '\n' + text
            input_text = ''
            input_text += self.template['INSTRUCTION'].format(input=text, round=1)
            input_text = past_text + input_text
            ids = self.tokenizer.encode(input_text)
            ids = torch.tensor(ids).to(self.device).unsqueeze(0)

            attention_mask = torch.ones_like(ids, dtype=torch.bool)

            mm_inputs = {
                'pixel_values': input_dict['pixel_values'].to(self.device),
                'input_ids': ids,
                'attention_mask': attention_mask,
            }

        generate_output = self.model.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        # sequences include the prompt for the input_ids-based generate
        generate_output_trimmed = generate_output.sequences[0][mm_inputs['input_ids'].shape[1]:]
        predict = self.tokenizer.decode(generate_output_trimmed, skip_special_tokens=False).strip()

        if image is None and video is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks, }

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        return {'prediction': predict, 'prediction_masks': ret_masks,}

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]
