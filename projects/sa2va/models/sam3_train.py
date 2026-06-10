"""SAM3 grounding encoder for Sa2VA, built from the vendored upstream SAM3 tracker.

This mirrors `sam2_train.SAM2TrainRunner` one-to-one so it is a drop-in
`grounding_encoder` for `Sa2VAModel` (same public surface: `hidden_dim`,
`sam2_model.sam_mask_decoder`, `preprocess_image`, `get_sam2_embeddings`,
`inject_language_embd`). The only change vs. SAM2 is the underlying segmentation
model: the SAM3 PVS tracker (`third_parts/sam3`, vendored from facebookresearch/sam3),
with the language-embedding injection done by `extension/sam3_base.Sam3Base`.

Checkpoint: SAM3 ships a single assembled `sam3.pt` (image/video multiplex model);
the standalone tracker weights live under sub-prefixes of it. `_load_sam3_checkpoint`
remaps those into this standalone tracker. See its docstring — the default prefix
map is a best guess from the upstream builder and may need adjustment once you can
inspect `sam3.pt` (set `SA2VA_SAM3_DUMP_KEYS=1` to print the top-level prefixes).
"""

import os

import torch

from mmengine.model import BaseModule

from projects.sa2va.models.extension import Sam3Base
from third_parts.sam3.build_sam3_tracker import build_sam3_tracker

# Default local checkpoint path (override via the runner's `ckpt_path`).
BASE_DIR = 'pretrained/sam3'


def _load_sam3_checkpoint(model, ckpt_path):
    """Load tracker weights from an assembled SAM3 checkpoint into `model`.

    The released `sam3.pt` is the assembled detector+tracker model; its tracker is
    built sharing the detector's vision backbone. Verified key layout (facebook/sam3):
      - tracker heads/memory/transformer  `tracker.*`                       -> `*`
      - shared vision backbone   `detector.backbone.vision_backbone.*` -> `backbone.vision_backbone.*`
    Everything else (detector transformer/heads, language backbone) is dropped.
    We strip DDP `module.`, remap, and load with strict=False, logging match counts.
    """
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    for k in ('model', 'model_state_dict', 'state_dict'):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break

    if os.environ.get('SA2VA_SAM3_DUMP_KEYS') == '1':
        prefixes = sorted({key.split('.')[0] for key in sd})
        print(f'[SAM3] checkpoint top-level prefixes: {prefixes}')

    # source-prefix -> dest-prefix (longest first so the more specific wins)
    remaps = [
        ('detector.backbone.vision_backbone.', 'backbone.vision_backbone.'),
        ('module.detector.backbone.vision_backbone.', 'backbone.vision_backbone.'),
        ('tracker.', ''),
        ('module.tracker.', ''),
    ]
    own_keys = set(model.state_dict().keys())
    new_sd = {}
    for key, val in sd.items():
        for src, dst in remaps:
            if key.startswith(src):
                cand = dst + key[len(src):]
                if cand in own_keys:
                    new_sd[cand] = val
                break

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    n_back = sum(1 for k in new_sd if k.startswith('backbone.'))
    n_dec = sum(1 for k in new_sd if k.startswith('sam_mask_decoder.'))
    print(f'[SAM3] loaded {len(new_sd)}/{len(own_keys)} tracker keys from {ckpt_path} '
          f'(backbone={n_back}, sam_mask_decoder={n_dec}); '
          f'missing={len(missing)}, unexpected={len(unexpected)}')
    if n_back == 0 or n_dec == 0:
        print('[SAM3] WARNING: backbone or mask-decoder matched 0 keys — the source '
              'prefix map is likely wrong for this checkpoint. Re-run with '
              'SA2VA_SAM3_DUMP_KEYS=1 and adjust `remaps` in _load_sam3_checkpoint.')


class Sam3TrackerTrainRunner(BaseModule):
    def __init__(
        self,
        ckpt_path: str = "sam3.pt",
        load_checkpoint: bool = True,
    ):
        super().__init__(init_cfg=None)

        # Build the standalone SAM3 PVS tracker with our language-conditioned base.
        sam3_model = build_sam3_tracker(base_cls=Sam3Base)

        if load_checkpoint:
            full_path = ckpt_path if os.path.isabs(ckpt_path) else os.path.join(BASE_DIR, ckpt_path)
            _load_sam3_checkpoint(sam3_model, full_path)

        # Keep the attribute name `sam2_model` so Sa2VAModel's hard-coded
        # `grounding_encoder.sam2_model.sam_mask_decoder` paths work unchanged.
        self.sam2_model = sam3_model

        self.hidden_dim = self.sam2_model.hidden_dim
        self.img_mean = (0.485, 0.456, 0.406)
        self.img_std = (0.229, 0.224, 0.225)

    # ------------------------------------------------------------------ #
    #  Image preprocessing (identical contract to SAM2TrainRunner)
    # ------------------------------------------------------------------ #
    def preprocess_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image / 255.
        img_mean = torch.tensor(self.img_mean, dtype=image.dtype, device=image.device)[:, None, None]
        img_std = torch.tensor(self.img_std, dtype=image.dtype, device=image.device)[:, None, None]
        image -= img_mean
        image /= img_std
        return image

    # ------------------------------------------------------------------ #
    #  Backbone features. `forward_image` returns SAM2-style backbone_fpn (BCHW);
    #  `_prepare_backbone_features` flattens to (HW, B, C) — same as SAM2.
    # ------------------------------------------------------------------ #
    def inject_language_embd(self, sam_states, language_embd, nf_nobj=None):
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(sam_states['current_vision_feats'][:-1], sam_states['feat_sizes'][:-1])
        ]

        B = sam_states['current_vision_feats'][-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = sam_states['feat_sizes'][-1]

        # Directly add the no-mem embedding (no memory attention for single-image grounding).
        pix_feat_with_mem = sam_states['current_vision_feats'][-1] + self.sam2_model.no_mem_embed
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, _, low_res_masks, high_res_masks, obj_ptr, _, = self.sam2_model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs=None,
                mask_inputs=None,
                high_res_features=high_res_features,
                multimask_output=self.sam2_model._use_multimask(is_init_cond_frame=True, point_inputs=None),
                # Inject language Embed if possible
                language_embd=language_embd,
            )

        if nf_nobj is not None:
            pred_masks = low_res_masks.squeeze(1)
            pred_masks = pred_masks.unflatten(0, nf_nobj)
        else:
            pred_masks = low_res_masks
        return pred_masks

    def get_sam2_embeddings(self, images, expand_size=1):
        # Step 1: run the frozen backbone with the images.
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = self.sam2_model.forward_image(images)

        if expand_size > 1:
            for i, feat in enumerate(feats["backbone_fpn"]):
                feats["backbone_fpn"][i] = feat[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
            for i, pos in enumerate(feats["vision_pos_enc"]):
                pos = pos[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
                feats["vision_pos_enc"][i] = pos

        # Step 2: Process the features to output
        _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.sam2_model._prepare_backbone_features(feats)

        return {
            "current_vision_feats": current_vision_feats,
            "current_vision_pos_embeds": current_vision_pos_embeds,
            "feat_sizes": feat_sizes,
        }

    def get_sam2_feats(self, images):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            feats = self.sam2_model.forward_image(images)
        return feats

    def get_sam2_states_from_feats(self, feats, expand_size=1):
        if expand_size > 1:
            for i, feat in enumerate(feats["backbone_fpn"]):
                feats["backbone_fpn"][i] = feat[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
            for i, pos in enumerate(feats["vision_pos_enc"]):
                pos = pos[:, None].expand(-1, expand_size, -1, -1, -1).flatten(0, 1)
                feats["vision_pos_enc"][i] = pos

        _, current_vision_feats, current_vision_pos_embeds, feat_sizes = self.sam2_model._prepare_backbone_features(feats)
        return {
            "current_vision_feats": current_vision_feats,
            "current_vision_pos_embeds": current_vision_pos_embeds,
            "feat_sizes": feat_sizes,
        }

    def forward(self, batch):
        raise NotImplementedError


# Looks-like-SAM2 alias.
SAM3TrainRunner = Sam3TrackerTrainRunner
