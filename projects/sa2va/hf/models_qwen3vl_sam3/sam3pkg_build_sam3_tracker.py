# Slim builder for the SAM3 PVS *tracker* (the SAM2-equivalent path), assembled
# WITH its own vision backbone so it can be used as a standalone grounding encoder
# in Sa2VA. The component recipes are copied verbatim from upstream
# `sam3/model_builder.py` (build_tracker / _create_vision_backbone / _create_tracker_*),
# trimmed to the tracker path and pointed at the vendored `third_parts.sam3` modules.
#
# Unlike SAM2 (hydra yaml), SAM3 is built in plain Python upstream, so we build in
# Python too. The `base_cls` hook lets Sa2VA inject its language-conditioned subclass
# (extension/sam3_base.py) in place of the vanilla `Sam3TrackerBase`.

from .sam3pkg_model_memory import (
    CXBlock,
    SimpleFuser,
    SimpleMaskDownSampler,
    SimpleMaskEncoder,
)
from .sam3pkg_model_model_misc import TransformerWrapper
from .sam3pkg_model_decoder import (
    TransformerDecoderLayerv2,
    TransformerEncoderCrossAttention,
)
from .sam3pkg_model_necks import Sam3DualViTDetNeck
from .sam3pkg_model_position_encoding import PositionEmbeddingSine
from .sam3pkg_model_sam3_tracker_base import Sam3TrackerBase
from .sam3pkg_model_vitdet import ViT
from .sam3pkg_model_vl_combiner import SAM3VLBackbone
from .sam3pkg_sam_transformer import RoPEAttention


def _create_position_encoding(precompute_resolution=None):
    return PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=precompute_resolution,
    )


def _create_vit_backbone(compile_mode=None, use_fa3=False, use_rope_real=False):
    return ViT(
        img_size=1008,
        pretrain_img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=32,
        num_heads=16,
        mlp_ratio=4.625,
        norm_layer="LayerNorm",
        drop_path_rate=0.1,
        qkv_bias=True,
        use_abs_pos=True,
        tile_abs_pos=True,
        global_att_blocks=(7, 15, 23, 31),
        rel_pos_blocks=(),
        use_rope=True,
        use_interp_rope=True,
        window_size=24,
        pretrain_use_cls_token=True,
        retain_cls_token=False,
        ln_pre=True,
        ln_post=False,
        return_interm_layers=False,
        bias_patch_embed=False,
        compile_mode=compile_mode,
        use_fa3=use_fa3,
        use_rope_real=use_rope_real,
    )


def _create_vit_neck(position_encoding, vit_backbone, enable_inst_interactivity=True):
    return Sam3DualViTDetNeck(
        position_encoding=position_encoding,
        d_model=256,
        scale_factors=[4.0, 2.0, 1.0, 0.5],
        trunk=vit_backbone,
        add_sam2_neck=enable_inst_interactivity,
    )


def _create_vision_backbone(compile_mode=None, enable_inst_interactivity=True):
    position_encoding = _create_position_encoding(precompute_resolution=1008)
    vit_backbone = _create_vit_backbone(compile_mode=compile_mode)
    vit_neck = _create_vit_neck(
        position_encoding, vit_backbone, enable_inst_interactivity=enable_inst_interactivity
    )
    return vit_neck


def _create_tracker_maskmem_backbone():
    position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
        precompute_resolution=1008,
    )
    mask_downsampler = SimpleMaskDownSampler(
        kernel_size=3, stride=2, padding=1, interpol_size=[1152, 1152]
    )
    cx_block_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1.0e-06,
        use_dwconv=True,
    )
    fuser = SimpleFuser(layer=cx_block_layer, num_layers=2)
    return SimpleMaskEncoder(
        out_dim=64,
        position_encoding=position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=fuser,
    )


def _create_tracker_transformer():
    self_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        use_fa3=False,
        use_rope_real=False,
    )
    cross_attention = RoPEAttention(
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
        rope_theta=10000.0,
        feat_sizes=[72, 72],
        rope_k_repeat=True,
        use_fa3=False,
        use_rope_real=False,
    )
    encoder_layer = TransformerDecoderLayerv2(
        cross_attention_first=False,
        activation="relu",
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pre_norm=True,
        self_attention=self_attention,
        d_model=256,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        cross_attention=cross_attention,
    )
    encoder = TransformerEncoderCrossAttention(
        remove_cross_attention_layers=[],
        batch_first=True,
        d_model=256,
        frozen=False,
        pos_enc_at_input=True,
        layer=encoder_layer,
        num_layers=4,
        use_act_checkpoint=False,
    )
    return TransformerWrapper(encoder=encoder, decoder=None, d_model=256)


def build_sam3_tracker(base_cls=Sam3TrackerBase, compile_mode=None):
    """Build a standalone SAM3 PVS tracker (with its own ViT backbone).

    Args:
        base_cls: the Sam3TrackerBase (sub)class to instantiate. Sa2VA passes its
            language-conditioned `Sam3Base` so `_forward_sam_heads` accepts `language_embd`.
    Returns:
        an instance of `base_cls` (a `Sam3TrackerBase`), randomly initialized.
        Weights are loaded separately by the caller from the SAM3 checkpoint.
    """
    maskmem_backbone = _create_tracker_maskmem_backbone()
    transformer = _create_tracker_transformer()
    vision_backbone = _create_vision_backbone(compile_mode=compile_mode)
    backbone = SAM3VLBackbone(scalp=1, visual=vision_backbone, text=None)

    model = base_cls(
        backbone=backbone,
        transformer=transformer,
        maskmem_backbone=maskmem_backbone,
        image_size=1008,
        backbone_stride=14,
        num_maskmem=7,
        # SAM head behavior (mirrors upstream build_tracker)
        multimask_output_in_sam=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,
        multimask_output_for_tracking=True,
        max_cond_frames_in_attn=4,
        sam_mask_decoder_extra_args={
            "dynamic_multimask_via_stability": True,
            "dynamic_multimask_stability_delta": 0.05,
            "dynamic_multimask_stability_thresh": 0.98,
        },
    )
    return model
