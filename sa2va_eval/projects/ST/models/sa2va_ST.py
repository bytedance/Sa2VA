import torch
import torch.nn as nn
import torch.nn.functional as F

from xtuner.registry import BUILDER

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.tools.utils import get_stop_criteria
from xtuner.model.utils import guess_load_checkpoint

from mmcv.ops import point_sample
from mmdet.models.utils import get_uncertain_point_coords_with_randomness

from mmengine.model import BaseModel
from projects.ST.dataset.utils import convert_image_to_patches, convert_mask_to_patches
from projects.ST.dataset.collect_fns import create_single_prefix_mask
from einops import rearrange
from transformers import DynamicCache, GenerationConfig
import copy
from mmengine.config import Config, ConfigDict
from peft import get_peft_model, prepare_model_for_kbit_training
import numpy as np
try:
    from third_parts.sam2.sam2_image_predictor import SAM2ImagePredictor
    from third_parts.sam2.sam2 import SAM2
except:
    pass

def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    if 'output_layer' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('output_layer')
    return list(lora_module_names)

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class DWconv(nn.Module):
    def __init__(self, nin, nout):
        super(DWconv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

NON_VISION_TOKEN = -1
PROMPT_TMPL = '<|im_start|>user\n{input}<|im_end|>\n'

class Sa2VASTModel(BaseModel):
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"
    def __init__(self,
                 single_transformer,
                 tokenizer,
                 single_transformer_lora=None,
                 seg_hidden_states=256,
                 patch_size=32,
                 seg_pred_down_ratio=4,
                 loss_mask=None,
                 loss_dice=None,
                 torch_dtype=torch.bfloat16,
                 pretrained_pth=None,
                 special_tokens=None,
                 visual_prompt_nums=10,
                 loss_sample_points=False,
                 num_points=12544,
                 # for inference
                 template=None,
                 add_cls=False,
                 bs=1,
                 # for distill
                 use_distill=False,
                 mask2former_model=None,
                 mask2former_processor=None,
                 sam2_model=None,
                 radio_processor=None,
                 radio_model=None,
                 distillation_loss_weight=1.0,
                 ):
        super().__init__()
        self.add_cls = add_cls
        self.bs = bs
        self.use_distill = use_distill
        self.patch_size = patch_size
        self.seg_pred_down_ratio = seg_pred_down_ratio
        self.seg_hidden_states = seg_hidden_states
        self.visual_prompt_nums = visual_prompt_nums
        visual_prompt_tokens = [f"<Prompt{i}>" for i in range(visual_prompt_nums)]
        if visual_prompt_nums != 0:
            visual_prompt_tokens.append("<NO_Prompt>")
        self.visual_prompt_tokens = visual_prompt_tokens
        if special_tokens is None:
            special_tokens = ['[SEG]']
        for visual_prompt_token in visual_prompt_tokens:
            assert visual_prompt_token in special_tokens
        self.special_tokens = special_tokens
        self.single_transformer = BUILDER.build(single_transformer)
        self.llm_hidden_states = self.single_transformer.config.hidden_size
        self.mask2former_model = mask2former_model
        if self.mask2former_model is not None and use_distill:
            assert mask2former_processor is not None
            self.mask2former_processor = BUILDER.build(mask2former_processor)
            self.mask2former_model = BUILDER.build(mask2former_model)
            self.mask2former_distill_linear = nn.Linear(seg_hidden_states, 256)
            self.mask2former_model.requires_grad_(False)
        else:
            self.mask2former_model = None

        self.sam2_model = sam2_model
        if self.sam2_model is not None and use_distill:
            sam2 = BUILDER.build(self.sam2_model)
            self._sam2 = sam2
            self.sam2_model = SAM2ImagePredictor(
                sam_model=sam2.sam2_model,
            )
            self.sam2_distill_linear = nn.Linear(self.llm_hidden_states, 256)
            self.sam2_model.model.requires_grad_(False)
        else:
            self.sam2_model = None

        self.radio_model = radio_model
        if self.radio_model is not None and use_distill:
            self.radio_model =  BUILDER.build(self.radio_model)
            self.radio_processor = BUILDER.build(radio_processor)

            self.radio_distill_linear = nn.Linear(self.llm_hidden_states, 1024)
            self.radio_model.model.requires_grad_(False)
        else:
            self.radio_model = None

        self.llm = self.single_transformer

        self.tokenizer = BUILDER.build(tokenizer)
        self._add_special_tokens()

        in_dim = min(self.single_transformer.config.hidden_size, 1024) # the hidden states of llm

        if in_dim == self.single_transformer.config.hidden_size:
            self.image_feature_pre_projector = None
        else:
            self.image_feature_pre_projector = nn.Sequential(
                nn.Linear(self.single_transformer.config.hidden_size, self.single_transformer.config.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(self.single_transformer.config.hidden_size, in_dim),
                nn.Dropout(0.0)
            )

        out_dim = seg_hidden_states
        self.seg_token_projector = nn.Sequential(
            nn.Linear(self.llm_hidden_states, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        upsample_block = nn.Sequential(
            nn.ConvTranspose2d(in_dim, in_dim, kernel_size=2, stride=2),
            nn.GELU(),
            DWconv(in_dim, in_dim),
            Norm2d(in_dim),
        )  # in, out (b, c, h , w)
        num_upsample_blocks = 1
        cur_stride = patch_size
        while cur_stride > 4:
            cur_stride /= 2
            num_upsample_blocks += 1
        # num_upsample_blocks = patch_size // 4
        self.upsample_blocks = [copy.deepcopy(upsample_block) for i in range(num_upsample_blocks)]
        self.upsample_blocks = nn.Sequential(*self.upsample_blocks)

        out_dim = seg_hidden_states
        self.image_feature_projector = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

        if single_transformer_lora is not None:
            self.single_transformer.requires_grad_(False)
            self.activation_checkpointing_enable()
            self.single_transformer.enable_input_require_grads()
            self._prepare_llm_for_lora(single_transformer_lora)
            self.single_transformer.model.base_model.get_input_embeddings().requires_grad_(True)
            self.single_transformer.lm_head.requires_grad_(True)

        self.loss_mask = BUILDER.build(loss_mask)
        self.loss_dice = BUILDER.build(loss_dice)

        self.distillation_loss_weight = distillation_loss_weight

        self.torch_dtype = torch_dtype

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            del_keys = []
            for key in pretrained_state_dict.keys():
                if "mask2former_model" in key or "mask2former_distill_linear" in key or \
                        "sam2_model" in key or "sam2_distill_linear" in key or \
                        "radio_model" in key or "radio_distill_linear" in key:
                    del_keys.append(key)
            for key in del_keys:
                del pretrained_state_dict[key]
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.loss_sample_points = loss_sample_points
        self.num_points = num_points
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.template = template
        self.template['INSTRUCTION'] = PROMPT_TMPL

        if visual_prompt_nums == 0:
            self.single_transformer.model.use_vp = False
        else:
            self.single_transformer.model.use_vp = True

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.single_transformer.model = prepare_model_for_kbit_training(
            self.single_transformer.model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.single_transformer.model)
            lora_config.target_modules = modules
        self.single_transformer.model = get_peft_model(self.single_transformer.model,
                                                   lora_config)

    def activation_checkpointing_disable(self):
        self.single_transformer.gradient_checkpointing_disable()

    def activation_checkpointing_enable(self):
        self.single_transformer.gradient_checkpointing_enable()

    def _add_special_tokens(self):

        self.tokenizer.vis_beg_tok = "<vision>"
        self.tokenizer.vis_patch_tok = "<vpatch>"
        self.tokenizer.vis_rsep_tok = "<vrow_sep>"
        self.tokenizer.vis_frm_tok = "<vframe_sep>"
        self.tokenizer.vis_end_tok = "</vision>"
        self.tokenizer.vis_cls_tok = "<|vis_cls|>"

        special_tokens = self.special_tokens
        _num_new_tokens = self.tokenizer.add_tokens(special_tokens, special_tokens=True)
        if _num_new_tokens > 0:
            print(f"Add {_num_new_tokens} tokens, resize the token embed.")
            self.single_transformer.resize_token_embeddings(len(self.tokenizer))
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.vision_patch_idx = self.tokenizer("<vpatch>", add_special_tokens=False).input_ids[0]

        self.vp_token_idxs = []
        for vp_token in self.visual_prompt_tokens:
            self.vp_token_idxs.append(self.tokenizer(vp_token, add_special_tokens=False).input_ids[0])
        # print("vp_token: ", self.vp_token_idxs)

        self.single_transformer.vp_token_idxs = self.vp_token_idxs
        self.single_transformer.model.vp_token_idxs = self.vp_token_idxs

    def state_dict(self, *args, **kwargs):
        self.llm = None
        state_dict = super().state_dict(*args, **kwargs)
        self.llm = self.single_transformer
        del_keys = []
        for key in state_dict.keys():
            if "mask2former_model" in key or "mask2former_distill_linear" in key or \
                    "sam2_model" in key or "sam2_distill_linear" in key or\
                    "radio_model" in key or "radio_distill_linear" in key:
                del_keys.append(key)
        for key in del_keys:
            del state_dict[key]
        return state_dict

    def _get_pesudo_data(self, device):
        gt_masks = torch.zeros((1, 256, 256), dtype=torch.uint8, device=device)
        gt_masks = [gt_masks] * self.bs
        return gt_masks

    def get_mask_prediction(self, seg_embeddings_list, image_seg_features):
        # seg_embedding (N, C)
        # image_feature (H, W, C)
        ret = []
        for seg_embeddings, image_seg_feature in zip(seg_embeddings_list, image_seg_features):
            pred_masks = torch.einsum("qc,hwc->qhw", seg_embeddings, image_seg_feature)
            ret.append(pred_masks)
        return ret

    def forward(self, data, data_samples=None, mode='loss'):
        images = data.pop('images', None)
        gt_masks = data.pop('masks', None)
        patch_nums_per_images = data.pop('patch_nums_per_images', None)
        input_ids = data['input_ids']

        if 'vision_patches' in data.keys() and data['vision_patches'] is not None:
            data['vision_patches'] = data['vision_patches'].flatten(1).to(self.torch_dtype)

        if gt_masks is None:
            # require zero seg datas
            seg_valid = False
            gt_masks = self._get_pesudo_data(
                device=input_ids.device,
            )
        else:
            seg_valid = True

        output = self.single_transformer(**data, return_dict=True, output_hidden_states=True)
        hidden_states = output.hidden_states
        # using last layer hidden states
        hidden_states = hidden_states[-1]

        # obtain image features
        image_token_mask = input_ids == self.vision_patch_idx
        vision_features = hidden_states[image_token_mask]  # (N, 256 * sub_pixels * sub_pixels)
        patch_split_nums = [item[0] * item[1] for item in patch_nums_per_images]
        vision_features = torch.split(vision_features, patch_split_nums, dim=0)
        all_image_features = []
        all_llm_image_features = []
        for patch_num, image_features in zip(patch_nums_per_images, vision_features):
            h_patches, w_patches = patch_num
            if h_patches * w_patches == 0:
                # no image
                # all_image_features.append(None)
                # using the first 100 tokens as pesudo image feature
                image_features = hidden_states[:1, :1].repeat(1, 100, 1).reshape(10, 10, self.llm_hidden_states).unsqueeze(
                    0)  # (1, h, w, c)
                all_llm_image_features.append(image_features)
                if self.image_feature_pre_projector is not None:
                    image_features = self.image_feature_pre_projector(image_features)
                image_features = image_features.permute(0, 3, 1, 2).contiguous()
                image_features = self.upsample_blocks(image_features)[0]  # (c, h, w)
                image_features = image_features.permute(1, 2, 0).contiguous()  # (h, w, c)
                image_features = self.image_feature_projector(image_features)
                all_image_features.append(image_features)
            else:
                image_features = image_features.reshape(h_patches, w_patches, self.llm_hidden_states).unsqueeze(0) # (1, h, w, c)
                all_llm_image_features.append(image_features)
                if self.image_feature_pre_projector is not None:
                    image_features = self.image_feature_pre_projector(image_features)
                image_features = image_features.permute(0, 3, 1, 2).contiguous()
                image_features = self.upsample_blocks(image_features)[0]  # (c, h, w)
                image_features = image_features.permute(1, 2, 0).contiguous() # (h, w, c)
                image_features = self.image_feature_projector(image_features)
                all_image_features.append(image_features)

        # obtain seg tokens
        seg_token_mask = input_ids == self.seg_token_idx
        if seg_valid:
            seg_token_features = self.seg_token_projector(hidden_states[seg_token_mask])
        else:
            seg_token_features = self.seg_token_projector(hidden_states[:, :1].flatten(0, 1))
        seg_token_counts = seg_token_mask.int().sum(-1)
        if not seg_valid:
            seg_token_counts += 1

        seg_embeddings_list_ = torch.split(seg_token_features, seg_token_counts.tolist(), dim=0)
        seg_embeddings_list = []
        image_seg_features = []
        gt_masks_ = []
        for idx, item in enumerate(seg_embeddings_list_):
            if len(item) != 0 and all_image_features[idx] is not None:
                seg_embeddings_list.append(item)
                image_seg_features.append(all_image_features[idx])
                gt_masks_.append(gt_masks[idx])
        gt_masks = gt_masks_

        if self.use_distill:
            assert images is not None
            distill_loss_dict = {}
            if self.mask2former_model is not None:
                distill_loss_dict.update(
                    self.compute_mask2former_distill_loss(images, all_image_features)
                )
            if self.sam2_model is not None:
                distill_loss_dict.update(
                    self.compute_sam2_distill_loss(images, all_llm_image_features)
                )
            if self.radio_model is not None:
                distill_loss_dict.update(
                    self.compute_radio_distill_loss(images, all_llm_image_features)
                )
        else:
            distill_loss_dict = {}

        pred_masks = self.get_mask_prediction(seg_embeddings_list, image_seg_features)
        if not self.loss_sample_points:
            gt_masks = [F.interpolate(gt_mask.unsqueeze(0), size=pred_mask.shape[-2:], mode='nearest').squeeze(0) for
                        gt_mask, pred_mask in zip(gt_masks, pred_masks)]

        loss_mask, loss_dice = 0, 0
        n_masks = 0
        for pred_mask, gt_mask in zip(pred_masks, gt_masks):
            # pred and gt mask, (n, h, w)
            if len(pred_mask) != len(gt_mask):
                # drop this data
                print(f"Pred mask shape {pred_mask.shape} is not equal to gt_mask shape {gt_mask.shape} !!!")
                min_num = min(len(pred_mask), len(gt_mask))
                pred_mask = pred_mask[:min_num]
                gt_mask = gt_mask[:min_num]
                _seg_valid = False
            else:
                _seg_valid = True

            if self.loss_sample_points:
                sampled_pred_mask, sampled_gt_mask = self.sample_points(pred_mask, gt_mask)
                sam_loss_dice = self.loss_dice(
                    sampled_pred_mask,
                    sampled_gt_mask, avg_factor=(1 + 1e-4))
                sam_loss_mask = self.loss_mask(
                    sampled_pred_mask.reshape(-1),
                    sampled_gt_mask.reshape(-1),
                    avg_factor=(sampled_pred_mask.shape[1] + 1e-4))
            else:
                sam_loss_mask = self.loss_mask(pred_mask, gt_mask) * len(pred_mask)
                sam_loss_dice = self.loss_dice(pred_mask, gt_mask) * len(pred_mask)

            if _seg_valid and seg_valid:
                _scale = 1.0
                n_masks += len(pred_mask)
            else:
                _scale = 0.0

            loss_mask += sam_loss_mask * _scale
            loss_dice += sam_loss_dice * _scale

        if loss_mask == 0.0:
            _llm_loss_scale = 1.0
        else:
            _llm_loss_scale = 0.1

        loss_dict = {
            'loss_mask': loss_mask / (n_masks + 1e-4) + output.loss * 0.0,
            'loss_dice': loss_dice / (n_masks + 1e-4) + output.loss * 0.0,
            'llm_loss': output.loss * _llm_loss_scale,
        }
        loss_dict.update(distill_loss_dict)
        return loss_dict

    def compute_mask2former_distill_loss(self, images, image_seg_features):
        assert len(images) == len(image_seg_features)
        device, dtype = image_seg_features[0].device, image_seg_features[0].dtype
        self.mask2former_model.eval()
        mask2former_pixel_features = []
        with torch.no_grad():
            for image in images:
                if image is None:
                    mask2former_pixel_features.append(None)
                else:
                    inputs = self.mask2former_processor(images=image, return_tensors="pt", size=800).to(dtype).to(device)
                    outputs = self.mask2former_model(**inputs, output_hidden_states=True)
                    mask2former_pixel_features.append(outputs.pixel_decoder_last_hidden_state)

        image_seg_features = [self.mask2former_distill_linear(image_seg_feature) for image_seg_feature in image_seg_features]
        loss = 0.0
        _image_num = 1e-4
        for mask2former_pixel_feature, image_seg_feature in zip(mask2former_pixel_features, image_seg_features):
            if mask2former_pixel_feature is None:
                loss += image_seg_feature.mean() * 0.0
            else:
                mask2former_pixel_feature = F.interpolate(
                    mask2former_pixel_feature, size=image_seg_feature.shape[:2], mode='bilinear'
                )[0].permute(1, 2, 0)  # (h, w, c)
                mse_loss = torch.mean((image_seg_feature - mask2former_pixel_feature) ** 2)
                # cos_sim = torch.nn.functional.cosine_similarity(
                #     image_seg_feature, mask2former_pixel_feature, dim=2
                # )
                # cos_loss = 1 - cos_sim.mean()
                # distill_loss = (0.9 * cos_loss + 0.1 * mse_loss) * self.distillation_loss_weight
                distill_loss = 1.0 * mse_loss * self.distillation_loss_weight
                loss += distill_loss
                _image_num += 1

        return {"mask2former_distill_loss": loss / _image_num}

    def compute_sam2_distill_loss(self, images, image_seg_features):
        assert len(images) == len(image_seg_features)
        device, dtype = image_seg_features[0].device, image_seg_features[0].dtype
        self.sam2_model.model.eval()
        sam2_features = []
        with torch.no_grad():
            for image in images:
                if image is None:
                    sam2_features.append(None)
                else:
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        self.sam2_model.set_image(image)
                        feature_dict = self.sam2_model._features
                        image_embed = feature_dict["image_embed"]
                    sam2_features.append(image_embed.to(dtype))

        image_seg_features = [self.sam2_distill_linear(image_seg_feature) for image_seg_feature in image_seg_features]
        loss = 0.0
        _image_num = 1e-4
        for sam2_feature, image_seg_feature in zip(sam2_features, image_seg_features):
            image_seg_feature = image_seg_feature[0]
            if sam2_feature is None:
                loss += image_seg_feature.mean() * 0.0
            else:
                sam2_feature = F.interpolate(
                    sam2_feature, size=image_seg_feature.shape[:2], mode='bilinear'
                )[0].permute(1, 2, 0)  # (h, w, c)
                mse_loss = torch.mean((image_seg_feature - sam2_feature) ** 2)
                # cos_sim = torch.nn.functional.cosine_similarity(
                #     image_seg_feature, mask2former_pixel_feature, dim=2
                # )
                # cos_loss = 1 - cos_sim.mean()
                # distill_loss = (0.9 * cos_loss + 0.1 * mse_loss) * self.distillation_loss_weight
                distill_loss = 1.0 * mse_loss * self.distillation_loss_weight
                loss += distill_loss
                _image_num += 1

        return {"sam2_distill_loss": loss / _image_num}

    def compute_radio_distill_loss(self, images, image_seg_features):
        assert len(images) == len(image_seg_features)
        device, dtype = image_seg_features[0].device, image_seg_features[0].dtype
        self.radio_model.eval()
        radio_features = []
        with torch.no_grad():
            for image in images:
                if image is None:
                    radio_features.append(None)
                else:
                    pixel_values = self.radio_processor(images=image, return_tensors='pt', do_resize=True).pixel_values
                    pixel_values = pixel_values.to(device)
                    nearest_res = self.radio_model.get_nearest_supported_resolution(*pixel_values.shape[-2:])
                    pixel_values = F.interpolate(pixel_values, nearest_res, mode='bilinear', align_corners=False)
                    pixel_values_shape = pixel_values.shape[-2:]
                    h_patch, w_patch = pixel_values_shape[0] // 16, pixel_values_shape[1] // 16
                    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                        summary, features = self.radio_model(pixel_values)
                        features = features.reshape(1, h_patch, w_patch, -1).permute(0, 3, 1, 2)
                    radio_features.append(features.to(dtype))

        image_seg_features = [self.radio_distill_linear(image_seg_feature) for image_seg_feature in image_seg_features]
        loss = 0.0
        _image_num = 1e-4
        for radio_feature, image_seg_feature in zip(radio_features, image_seg_features):
            image_seg_feature = image_seg_feature[0]
            if radio_feature is None:
                loss += image_seg_feature.mean() * 0.0
            else:
                radio_feature = F.interpolate(
                    radio_feature, size=image_seg_feature.shape[:2], mode='bilinear'
                )[0].permute(1, 2, 0)  # (h, w, c)
                mse_loss = torch.mean((image_seg_feature - radio_feature) ** 2)
                # cos_sim = torch.nn.functional.cosine_similarity(
                #     image_seg_feature, mask2former_pixel_feature, dim=2
                # )
                # cos_loss = 1 - cos_sim.mean()
                # distill_loss = (0.9 * cos_loss + 0.1 * mse_loss) * self.distillation_loss_weight
                distill_loss = 1.0 * mse_loss * self.distillation_loss_weight
                loss += distill_loss
                _image_num += 1

        return {"radio_distill_loss": loss / _image_num}

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        mask_pred = mask_pred.unsqueeze(1)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred.to(torch.float32), None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
        return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)

    def preparing_for_generation(self, metainfo, **kwargs):
        # set stop criteria and generation configs for model
        assert hasattr(self, 'tokenizer'), "The Model does not have the tokenizer!!!"
        self.bot_name = 'BOT'
        if 'template' in metainfo.keys():
            template = metainfo['template']
        else:
            template = PROMPT_TEMPLATE['phi3_chat']
        if self.template is None:
            self.template = template
        stop_words = []
        stop_words += self.template.get('STOP_WORDS', [])
        stop_criteria = get_stop_criteria(
            tokenizer=self.tokenizer, stop_words=stop_words)
        self.stop_criteria = stop_criteria

        default_generation_kwargs = dict(
            max_new_tokens=512,
            do_sample=False,
            temperature=0,
            num_beams=1,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        default_generation_kwargs.update(metainfo.get('generation_kwargs', {}))
        self.gen_config = GenerationConfig(**default_generation_kwargs)
        self.init_prediction_config = True

        self.single_transformer.to(self.torch_dtype)
        self.seg_token_projector.to(self.torch_dtype)
        self.image_feature_projector.to(self.torch_dtype)
        self.upsample_blocks.to(self.torch_dtype)
        if self.image_feature_pre_projector is not None:
            self.image_feature_pre_projector.to(self.torch_dtype)
        return

    def prepare_image_textual_seq_norowsep(self, h, w):
        image_token_patch_indices = []
        seq = ""
        tok_len = 0

        seq += self.IMG_START_TOKEN
        tok_len += 1
        image_token_patch_indices.append(NON_VISION_TOKEN)

        seq += self.IMG_CONTEXT_TOKEN * (w * h)
        tok_len += (w * h)
        image_token_patch_indices += [idx for idx in range(w * h)]

        seq += self.IMG_END_TOKEN
        tok_len += 1
        image_token_patch_indices.append(NON_VISION_TOKEN)

        if self.add_cls:
            seq += self.CLS_TOKEN
            tok_len += 1
            image_token_patch_indices.append(NON_VISION_TOKEN)
        return seq, tok_len, image_token_patch_indices

    def predict_forward(
            self,
            image=None,
            text=None,
            past_text='',
            prompt_masks=None,
            prompt_ids=None,
            prefix_answer=None,
    ):
        assert self.tokenizer
        if prompt_ids is not None:
            for _id in prompt_ids:
                assert 0 <= _id < self.visual_prompt_nums

        input_dict = {}
        ori_image_size = image.size

        if image is None:
            input_dict['vision_patches'] = None
            input_dict['patch_nums_per_images'] = (0, 0)
            image_token_str = ''
            image_token_patch_indices = []
            vp_token_indices = []
        else:
            image_patches = convert_image_to_patches(image, self.patch_size)
            if prompt_masks is not None:
                masks_prompt_patches = [convert_mask_to_patches(mask, self.patch_size) for mask in prompt_masks]
            else:
                masks_prompt_patches = None
            # tensor, (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
            h_patches, w_patches = image_patches.shape[:2]
            n_patches = h_patches * w_patches
            # input_dict['vision_patches'] = image_patches.view(n_patches, -1)  # (n_patches, 3*patch_size*patch_size)
            input_dict['vision_patches'] = image_patches.flatten(0, 1).flatten(1)  # (n_patches, 3*patch_size*patch_size)
            input_dict['patch_nums_per_images'] = (h_patches, w_patches)
            image_token_str, image_token_len, image_token_patch_indices, vp_token_indices = \
                self.prepare_image_textual_seq_norowsep_with_vp(
                    image_patches.shape[0], image_patches.shape[1], masks=masks_prompt_patches, vp_tokens=prompt_ids,
                )
            # print(vp_token_indices)
        ret_masks = []
        if '<image>' in text:
            assert past_text is None or len(past_text) == 0
            first_conv = True
        else:
            first_conv = False
        text = text.replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
        input_text = ''
        input_text += self.template['INSTRUCTION'].format(
                input=text, round=1, bot_name=self.bot_name)
        if first_conv:
            input_text = image_token_str + input_text
        else:
            input_text = past_text + input_text
        if prefix_answer is not None:
            input_text = input_text + prefix_answer

        ids = self.tokenizer.encode(input_text, add_special_tokens=False)
        _pre_length = len(ids)
        vision_start_end = self.search_vision_tokens(ids)

        attention_mask = create_single_prefix_mask(vision_start_end, len(ids)).unsqueeze(0).unsqueeze(0).cuda()
        # attention_mask = create_single_prefix_mask(vision_start_end, len(ids)).unsqueeze(0).cuda()

        ids = torch.tensor(ids).cuda().unsqueeze(0)
        position_ids = generate_mm_pos_ids_singleit(
            ids[0].cpu().numpy().tolist(), self.vision_patch_idx,
            input_dict['patch_nums_per_images'][0], input_dict['patch_nums_per_images'][1]).unsqueeze(1).cuda()

        vision_patch_indices = []
        visual_prompt_token_indices = []
        vision_patch_indices += image_token_patch_indices
        visual_prompt_token_indices += vp_token_indices
        vision_patch_indices += [NON_VISION_TOKEN] * (ids.shape[-1] - len(vision_patch_indices))
        visual_prompt_token_indices += [NON_VISION_TOKEN] * (ids.shape[-1] - len(visual_prompt_token_indices))

        vision_patch_indices = torch.tensor(vision_patch_indices).cuda().unsqueeze(0)
        visual_prompt_token_indices = torch.tensor(visual_prompt_token_indices).cuda().unsqueeze(0)

        vp_token_indices = visual_prompt_token_indices

        padding_attention_mask = torch.ones_like(ids).cuda()

        mm_inputs = {
            'vision_patches': input_dict['vision_patches'].flatten(1).cuda().to(self.torch_dtype),
            # 'vision_patches': None,
            'input_ids': ids,
            'attention_mask': padding_attention_mask,
            'position_ids': position_ids,
            'labels': None,
            'vision_patch_indices': vision_patch_indices,
            'vp_token_indices': vp_token_indices,
        }

        # first forward for none casual image tokens
        image_tokens_len = vision_start_end[-1] + 1
        cached_inputs = dict(
            input_ids=ids[:, :image_tokens_len],
            position_ids=position_ids[:, :, :image_tokens_len],
            attention_mask=attention_mask[:, :, :image_tokens_len, :image_tokens_len],
            vision_patches=mm_inputs['vision_patches'],
            vision_patch_indices=vision_patch_indices[:, :image_tokens_len],
            vp_token_indices=vp_token_indices[:, :image_tokens_len],
            use_cache=True
        )
        prefix_cache = DynamicCache()
        with torch.no_grad():
            prefix_cache = self.single_transformer.forward(**cached_inputs, past_key_values=prefix_cache,
                                                           return_dict=True, output_hidden_states=True)
            past_hidden_states = prefix_cache.hidden_states
            prefix_cache = prefix_cache.past_key_values
        past_key_values = copy.deepcopy(prefix_cache)

        generate_output = self.single_transformer.generate(
            **mm_inputs,
            generation_config=self.gen_config,
            streamer=None,
            bos_token_id=self.tokenizer.bos_token_id,
            stopping_criteria=self.stop_criteria,
            output_hidden_states=True,
            return_dict_in_generate=True,
            past_key_values=past_key_values,
        )
        predict = self.tokenizer.decode(
            generate_output.sequences[0][_pre_length:], skip_special_tokens=False).strip()
        # print(self.tokenizer.decode(
        #     generate_output.sequences[0][_pre_length-20:], skip_special_tokens=False).strip())

        # past key tokens
        last_past_hidden_states = past_hidden_states[-1][0]

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)

        last_hidden_states = torch.cat([last_past_hidden_states, last_hidden_states], dim=0)

        # obtain image features
        image_token_mask = ids[0] == self.vision_patch_idx
        vision_features = last_hidden_states[:len(ids[0])][image_token_mask]  # (N, c)
        patch_split_nums = [item[0] * item[1] for item in [input_dict['patch_nums_per_images']]]
        vision_features = torch.split(vision_features, patch_split_nums, dim=0)
        all_image_features = []
        for patch_num, image_features in zip([input_dict['patch_nums_per_images']], vision_features):
            h_patches, w_patches = patch_num
            if h_patches * w_patches == 0:
                # no image
                all_image_features.append(None)
            else:
                image_features = image_features.reshape(h_patches, w_patches, self.llm_hidden_states).unsqueeze(
                    0)  # (1, h, w, c)
                if self.image_feature_pre_projector is not None:
                    image_features = self.image_feature_pre_projector(image_features)
                image_features = image_features.permute(0, 3, 1, 2).contiguous()
                image_features = self.upsample_blocks(image_features)[0]  # (c, h, w)
                image_features = image_features.permute(1, 2, 0).contiguous()  # (h, w, c)
                image_features = self.image_feature_projector(image_features)
                all_image_features.append(image_features)
        image_features = all_image_features[0]

        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.seg_token_projector(seg_hidden_states)
        if all_seg_hidden_states.shape[0] == 0:
            ret_masks = None
        else:
            pred_masks = torch.einsum("qc,hwc->qhw", all_seg_hidden_states, image_features)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)[0]
            masks = masks.sigmoid() > 0.5
            # masks = masks.cpu().numpy()
            masks = masks.cpu()
            ret_masks.append(masks)
        return {'prediction': predict, 'prediction_masks': ret_masks, 'input_text': ''}

    def search_vision_tokens(self, input_ids):
        image_start_idx = self.tokenizer(self.IMG_START_TOKEN, add_special_tokens=False).input_ids[0]
        image_end_idx = self.tokenizer(self.IMG_END_TOKEN, add_special_tokens=False).input_ids[0]
        if image_start_idx not in input_ids:
            return None
        else:
            start_idx = input_ids.index(image_start_idx)
            end_idx = input_ids.index(image_end_idx)
            return [start_idx+1, end_idx]

    def prepare_image_textual_seq_norowsep_with_vp(self, h, w, masks=None, vp_tokens=None):
        image_token_patch_indices = []
        vp_token_indices = []
        seq = ""
        tok_len = 0

        seq += self.IMG_START_TOKEN
        tok_len += 1
        image_token_patch_indices.append(NON_VISION_TOKEN)
        vp_token_indices.append(NON_VISION_TOKEN)

        seq += self.IMG_CONTEXT_TOKEN * (w * h)
        tok_len += (w * h)
        image_token_patch_indices += [idx for idx in range(w * h)]

        _vision_vp_tokens = np.zeros((w*h), dtype=np.int64) - 2  # -2 for none vp index
        if masks is not None:
            assert len(masks) == len(vp_tokens)
            for mask, vp_index in zip(masks, vp_tokens):
                mask = mask.flatten(0, 1).bool().numpy()
                _vision_vp_tokens[mask] = vp_index
        _vision_vp_tokens = _vision_vp_tokens.tolist()
        vp_token_indices += _vision_vp_tokens

        seq += self.IMG_END_TOKEN
        tok_len += 1
        image_token_patch_indices.append(NON_VISION_TOKEN)
        vp_token_indices.append(NON_VISION_TOKEN)

        if self.add_cls:
            seq += self.CLS_TOKEN
            tok_len += 1
            image_token_patch_indices.append(NON_VISION_TOKEN)
            vp_token_indices.append(NON_VISION_TOKEN)
        return seq, tok_len, image_token_patch_indices, vp_token_indices

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]


def generate_mm_pos_ids_singleit(input_ids, vpatch_id, h, w):
    input_ids_pt = torch.Tensor(input_ids).int()
    vpatch_pos = torch.argwhere(input_ids_pt == vpatch_id)
    vpatch_start_pos = vpatch_pos[0].item()
    nt = len(input_ids) - (h * w) + 1

    # v_pos
    t_indices = torch.arange(1)
    h_indices = torch.arange(h)
    w_indices = torch.arange(w)
    v_pos_id = torch.stack(torch.meshgrid(t_indices, h_indices, w_indices, indexing='ij'), dim=0)
    v_pos_id = rearrange(v_pos_id, "d t h w -> (t h w) d")  # [h*w, 3]
    v_pos_id += vpatch_start_pos
    position_id = torch.cat(
        [
            torch.arange(vpatch_start_pos).unsqueeze(-1).repeat(1, 3),
            v_pos_id,
            torch.arange(nt - vpatch_start_pos - 1).unsqueeze(-1).repeat(1, 3) + v_pos_id.max() + 1,
        ],
        dim=0
    )
    assert len(input_ids) == position_id.size(0)
    position_id = rearrange(position_id, "slen d -> d slen").long()

    return position_id
