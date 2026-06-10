"""SAM3 grounding encoder wrapper for the HF Sa2VA model (inference).

This mirrors the SAM2 HF wrapper (`../models_qwen3vl/sam2.py`, class `SAM2`) so it
is a drop-in `grounding_encoder` for the HF `Sa2VAChatModelQwen`: same public
surface (`hidden_dim`, `image_size`, `preprocess_image`, `get_sam2_embeddings`,
`inject_language_embd`, `language_embd_inference`) and the attribute name
`sam2_model` so the parent model's hard-coded paths keep working.

Unlike SAM2 (whose model code is fully vendored inline into `sam2.py`), the SAM3
tracker is large and already vendored under `third_parts/sam3`. We therefore build
on top of it directly. The video-predictor inference path (`init_state`,
`add_language_embd`, `propagate_in_video`) lives in
`third_parts/sam3/model/sam3_tracking_predictor.py`; the language-conditioned mask
head (`_forward_sam_heads` with `language_embd`) lives in
`projects/sa2va/models/extension/sam3_base.py`. We combine them via MRO so a single
class is both a video predictor and language-conditioned. The standalone tracker
is built by `build_sam3_tracker` exactly as in training, so the parameter names
(and hence the converted state-dict keys) match one-to-one.
"""

import torch
import torch.nn as nn

from . import sam3pkg_build_sam3_tracker as _build_mod
from .sam3pkg_build_sam3_tracker import build_sam3_tracker
from .sam3pkg_model_memory import CXBlock as _CXBlock
from .sam3pkg_model_sam3_tracking_predictor import Sam3TrackerPredictor
from .sam3pkg_ext_sam3_base import Sam3Base


class _CXBlockGWeight(_CXBlock):
    """CXBlock whose layer-scale parameter is named `g_weight` instead of `gamma`.

    transformers' `from_pretrained` auto-renames any checkpoint key ending in
    `.gamma` -> `.weight` (and `.beta` -> `.bias`), which would corrupt SAM3's
    ConvNeXt layer-scale on load. The SAM2 HF model dodges this by naming the
    parameter `g_weight`; we mirror that here so the converted SAM3 checkpoint
    (whose `.gamma` keys are remapped to `.g_weight` by `tools/convert_to_hf.py`)
    loads cleanly and is not silently renamed at eval time.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        gamma = self._parameters.pop('gamma', None)
        if gamma is not None:
            self.register_parameter('g_weight', gamma)
        else:
            self.g_weight = None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.g_weight is not None:
            x = self.g_weight * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x


# The builder binds `CXBlock` at import time, so patch the name in its namespace
# (patching `memory.CXBlock` would not affect the already-imported reference).
_build_mod.CXBlock = _CXBlockGWeight


class Sam3LangPredictor(Sam3TrackerPredictor, Sam3Base):
    """SAM3 video predictor whose SAM mask heads accept an LLM `[SEG]` embedding.

    MRO: Sam3LangPredictor -> Sam3TrackerPredictor (video-state methods) ->
    Sam3Base (`_forward_sam_heads` with `language_embd`) -> Sam3TrackerBase.
    """

    pass


class SAM3(nn.Module):
    def __init__(self, ckpt_path: str = None):
        super().__init__()

        # Build the standalone SAM3 PVS tracker with the language-conditioned,
        # video-predictor base. Weights are loaded by the parent model's
        # `load_state_dict`, so we never load a checkpoint here.
        sam2_model = build_sam3_tracker(base_cls=Sam3LangPredictor)

        # Keep the attribute name `sam2_model` so the parent Sa2VA model's
        # hard-coded `grounding_encoder.sam2_model.*` paths work unchanged.
        self.sam2_model = sam2_model

        self.hidden_dim = self.sam2_model.hidden_dim
        self.image_size = self.sam2_model.image_size  # SAM3 native: 1008

        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

    def get_sam2_embeddings(self, images):
        # Build an inference state directly from a preprocessed image tensor
        # (shape: [num_frames, 3, image_size, image_size]). The SAM3 predictor's
        # `init_state` only populates `images` from a video path, so we set the
        # explicit dimensions and inject the tensor (mirrors SAM2 HF `init_state`).
        inference_state = self.sam2_model.init_state(
            video_height=self.image_size,
            video_width=self.image_size,
            num_frames=len(images),
        )
        inference_state["images"] = images
        return inference_state

    def inject_language_embd(self, inference_state, language_embd):
        num_frame = len(language_embd)
        num_obj = len(language_embd[0])
        mask_out = []
        for frame_idx in range(num_frame):
            frame_mask_out = []
            for obj_idx in range(num_obj):
                _language_embd = language_embd[frame_idx][obj_idx][None][None]
                _, _, out_mask_logits = self.sam2_model.add_language_embd(
                    inference_state, frame_idx, obj_idx + 100, _language_embd
                )
                frame_mask_out.append(out_mask_logits)
            frame_mask_out = torch.cat(frame_mask_out, dim=1)
            mask_out.append(frame_mask_out)
        mask_out = torch.cat(mask_out, dim=0)
        return mask_out

    @torch.no_grad()
    def language_embd_inference(self, inference_state, language_embd):
        # SAM3's ViT uses a fused perflib kernel that asserts grad is disabled, so
        # the whole grounding-encoder forward must run under no_grad.
        num_frame = len(language_embd)
        num_obj = len(language_embd[0])
        mask_out = []
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for frame_idx in range(num_frame):
                for obj_idx in range(num_obj):
                    _language_embd = language_embd[frame_idx][obj_idx][None][None]
                    self.sam2_model.add_language_embd(
                        inference_state,
                        frame_idx,
                        obj_idx + 100,
                        _language_embd,
                        inference=True,
                    )

            mask_out = []
            # SAM3's `propagate_in_video` requires explicit tracking args and yields
            # (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores); we take
            # the video-resolution masks (index 3) to match SAM2's contract.
            for out in self.sam2_model.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=inference_state["num_frames"],
                reverse=False,
                propagate_preflight=True,
            ):
                out_mask_logits = out[3]
                mask_out.append(out_mask_logits)
            mask_out = torch.cat(mask_out, dim=0)
        return mask_out

    def get_sam2_embeddings_with_expand(self, images, expand_size=1):
        raise NotImplementedError

    def forward(self, batch):
        raise NotImplementedError

    def preprocess_image(self, image: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        image = image / 255.

        img_mean = torch.tensor(self.img_mean, dtype=dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std

        return image
