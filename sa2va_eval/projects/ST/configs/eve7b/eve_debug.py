from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from mmdet.models import DiceLoss, CrossEntropyLoss

from projects.ST.models.sa2va_ST_eve import Sa2VASTEVEModel
import torch
from projects.ST.hooks.evaluation_chat_hook import EvaluateChatHook_ST
from projects.ST.dataset.eve.vqa_dataset import LLaVADataset
from projects.ST.dataset.eve.PixeCapVisualPrompt_Dataset import PixelCapVisualPromptDataset
from projects.ST.dataset.eve.RefCOCO_Dataset import ReferSegmDataset
from projects.ST.dataset.eve.collect_fns_eve import st_eve_collate_fn
from projects.ST.eve.model import EVEQwen2ForCausalLM
from transformers import CLIPImageProcessor

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
mllm_path = "pretrained/eve/EVE-7B-HD-v2.0"
# pretrained_pth = "pretrained/eve/eve.pth"
pretrained_pth = None

# Data
prompt_template = PROMPT_TEMPLATE.qwen_chat
max_length = 8192

vision_patch_size = 32

# Scheduler & Optimizer
batch_size = 1  # per_device
accumulative_counts = 32
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
# official 1024 -> 4e-5
# lr = 1e-6
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1  # grad clip
warmup_ratio = 0.05

# Save
save_steps = 5000
save_total_limit = 2  # Maximum checkpoints to keep (-1 means unlimited)

visual_prompt_nums = 15
visual_prompt_tokens = [f"<Prompt{i}>" for i in range(visual_prompt_nums)]
visual_prompt_tokens.append("<NO_Prompt>")

special_tokens = ['[SEG]', "<p>", "</p>"] + visual_prompt_tokens

evaluation_freq = 500
evaluation_images = './test.jpg'
evaluation_inputs = ['Please describe this picture.']

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=mllm_path,
    trust_remote_code=True,
    use_fast=True,
    padding_side='right')

# image_preprocessor = dict(
#     type=CLIPImageProcessor.from_pretrained,
#     pretrained_model_name_or_path="openai/eve-anyratio-res1600-patch16",
# )
image_preprocessor = dict(
    type=CLIPImageProcessor.from_pretrained,
    pretrained_model_name_or_path="openai/eve-anyratio-res800-patch16",
)


#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
model = dict(
    type=Sa2VASTEVEModel,
    single_transformer=dict(
        type=EVEQwen2ForCausalLM.from_pretrained,
        pretrained_model_name_or_path=mllm_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
        # use_cache=False,
        # low_cpu_mem_usage=True,
    ),
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    seg_hidden_states=256,
    patch_size=vision_patch_size,
    seg_pred_down_ratio=4,
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    torch_dtype=torch.bfloat16,
    pretrained_pth=pretrained_pth,
    # pretrained_pth=None,
    loss_sample_points=True,
    num_points=12544,
    # for inference
    template=prompt_template,
    bs=batch_size,
    visual_prompt_nums=visual_prompt_nums,
    # for eval
    use_eval=True,
    eval_interval=500,
    eval_image=evaluation_images,
    eval_text='<image>\nPlease describe this picture.',
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

################## image chat
llava_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    image_preprocessor=image_preprocessor,
    data_path='data/llava_data/LLaVA-Instruct-150K/llava_v1_5_mix665k.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    image_folder='data/llava_data/llava_images/',
    max_length=max_length,
    patch_size=vision_patch_size,
)

refcoco_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    image_preprocessor=image_preprocessor,
    special_tokens=special_tokens,
    data_root='data/ref_seg/refcoco',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    patch_size=vision_patch_size,
)
refcoco_plus_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    image_preprocessor=image_preprocessor,
    special_tokens=special_tokens,
    data_root='data/ref_seg/refcoco+',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    patch_size=vision_patch_size,
)
refcocog_segm_dataset=dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    image_preprocessor=image_preprocessor,
    special_tokens=special_tokens,
    data_root='data/ref_seg/refcocog',
    data_prefix=dict(img_path='coco2014/train2014/'),
    ann_file='instances.json',
    split_file='refs(umd).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    patch_size=vision_patch_size,
)

# coco_pano_segm_dataset=dict(
#     type=COCOSegmDataset,
#     image_folder="data/coco/Images/train2017",
#     pano_rgb_folder="data/coco/Annotations/annotations/panoptic_train2017/",
#     pano_json="data/coco/Annotations/annotations/panoptic_train2017.json",
#     class_txt="projects/ST/dataset/coco_panoptic_with_prompt_eng.txt",
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     num_classes_per_sample=5,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
# )
#
# coco_pano_semantic_segm_dataset=dict(
#     type=COCOSemanticSegmDataset,
#     image_folder="data/coco/Images/train2017",
#     pano_rgb_folder="data/coco/Annotations/annotations/panoptic_train2017/",
#     pano_json="data/coco/Annotations/annotations/panoptic_train2017.json",
#     class_txt="projects/ST/dataset/coco_panoptic_with_prompt_eng.txt",
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     num_classes_per_sample=5,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
# )
#
# pixelcap_segm_dataset=dict(
#     type=PixelCapSegDataset,
#     image_folder="data/coco/Images/train2017",
#     pano_rgb_folder="data/coco/Annotations/annotations/panoptic_train2017/",
#     pano_json="data/pixel2cap/pix2cap_coco_train.json",
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     num_classes_per_sample=5,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
# )
#
# # image gcg datas
# glamm_data_root = './data/glamm_data/'
#
# refcocog_image_path = glamm_data_root + 'images/coco2014/train2014/'
# refcocog_ann_file = glamm_data_root + 'annotations/RefCOCOg_GCG_train.json'
#
# grandf_image_path = glamm_data_root + 'images/grandf/train/'
# grandf_ann_file = glamm_data_root + 'annotations/GranDf_HA_GCG_train.json'
#
# flickr_image_path = glamm_data_root + 'images/flickr30k/Flickr30K/'
# flickr_ann_file = glamm_data_root + 'annotations/flickr_mergedGT_GCG_train.json'
#
# psg_image_path = glamm_data_root + 'images/coco2017/'
# psg_ann_file = glamm_data_root + 'annotations/OpenPsgGCG_train.json'
#
# glamm_grandf_dataset=dict(
#     type=GlammGrandfDataset,
#     image_folder=grandf_image_path,
#     data_path=grandf_ann_file,
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
#     repeats=10,
# )
#
# glamm_refcocog_dataset=dict(
#     type=GlammRefcocogDataset,
#     image_folder=refcocog_image_path,
#     data_path=refcocog_ann_file,
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
#     repeats=1,
# )
#
# glamm_psg_dataset=dict(
#     type=GlammOpenpsgDataset,
#     image_folder=psg_image_path,
#     data_path=psg_ann_file,
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
#     repeats=1,
# )
#
# glamm_flicker_dataset=dict(
#     type=GlammFlickerDataset,
#     image_folder=flickr_image_path,
#     data_path=flickr_ann_file,
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
#     repeats=1,
# )
#
# muse_dataset=dict(
#     type=MuseDataset,
#     train2014_image_folder="data/glamm_data/images/coco2014/train2014/",
#     train2017_image_folder="data/coco/Images/train2017/",
#     val2017_image_folder="data/coco/Images/val2017/",
#     json_file="data/muse/MUSE_train.json",
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     num_classes_per_sample=5,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
# )
#
# gref_coco_segm_dataset = dict(
#     type=GRefCoCoDataset,
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     data_root='data/ref_seg/grefs',
#     data_prefix=dict(img_path='coco2014/train2014/'),
#     ann_file='instances.json',
#     split_file='grefs(unc).json',
#     prompt_template=prompt_template,
#     num_classes_per_sample=5,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
# )

pixelcap_vp_dataset=dict(
    type=PixelCapVisualPromptDataset,
    image_folder="data/coco/Images/train2017",
    pano_rgb_folder="data/coco/Annotations/annotations/panoptic_train2017/",
    pano_json="data/pixel2cap/pix2cap_coco_train.json",
    tokenizer=tokenizer,
    image_preprocessor=image_preprocessor,
    special_tokens=special_tokens,
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    max_length=max_length,
    patch_size=vision_patch_size,
    prompt_numbers=visual_prompt_nums,
    repeats=5,
)

# # osprey
# data_osprey_file = 'data/osprey-724k/osprey_conversation.json'
# data_osprey_image_folders = [
#     'data/glamm_data/images/coco2014/train2014/',
#     'data/coco2014/val2014/',
#     'data/coco/train2017/',
#     'data/coco/val2017/',
# ]
#
# osprey_conversation_dataset = dict(
#     type=OspreyDataset,
#     image_folder=data_osprey_image_folders,
#     data_path=data_osprey_file,
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     num_classes_per_sample=5,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
#     prompt_numbers=visual_prompt_nums,
# )
#
# data_osprey_detail_description_file = 'data/osprey-724k/osprey_detail_description.json'
# osprey_description_dataset = dict(
#     type=OspreyDescriptionDataset,
#     image_folder=data_osprey_image_folders,
#     data_path=data_osprey_detail_description_file,
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     num_classes_per_sample=5,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
#     prompt_numbers=visual_prompt_nums,
#     repeats=10,
# )
#
# data_osprey_short_file = 'data/osprey-724k/osprey_short_form.json'
# osprey_short_description_dataset = dict(
#     type=OspreyShortDescriptionDataset,
#     image_folder=data_osprey_image_folders,
#     data_path=data_osprey_short_file,
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     num_classes_per_sample=5,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
#     prompt_numbers=visual_prompt_nums,
# )
#
# data_osprey_part_file = 'data/osprey-724k/osprey_part_level.json'
# osprey_part_dataset = dict(
#     type=OspreyDataset,
#     image_folder=data_osprey_image_folders,
#     data_path=data_osprey_part_file,
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     num_classes_per_sample=5,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
#     prompt_numbers=visual_prompt_nums,
# )
#
# data_osprey_positive_neg_file = 'data/osprey-724k/osprey_lvis_positive_negative.json'
# osprey_positive_neg_dataset = dict(
#     type=OspreyDataset,
#     image_folder=data_osprey_image_folders,
#     data_path=data_osprey_positive_neg_file,
#     tokenizer=tokenizer,
#     special_tokens=special_tokens,
#     prompt_template=prompt_template,
#     num_classes_per_sample=5,
#     max_length=max_length,
#     patch_size=vision_patch_size,
#     add_cls=False,
#     prompt_numbers=visual_prompt_nums,
# )

train_dataset = dict(
    type=ConcatDataset, datasets=[
        # coco
        # coco_pano_semantic_segm_dataset, coco_pano_segm_dataset,
        # glamm
        # glamm_psg_dataset, glamm_flicker_dataset, glamm_grandf_dataset, glamm_refcocog_dataset,
        # muse
        # muse_dataset,
        # pixel2cap
        # pixelcap_segm_dataset, pixelcap_segm_dataset,
        # ref seg
        refcoco_segm_dataset,
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # gres
        # gref_coco_segm_dataset, gref_coco_segm_dataset, gref_coco_segm_dataset, gref_coco_segm_dataset,
        # image qa
        # llava_vqa_dataset,
        # pixel2cap vp
        # pixelcap_vp_dataset,
        # osprey
        # osprey_description_dataset, osprey_short_description_dataset, osprey_conversation_dataset,
        # osprey_part_dataset, osprey_positive_neg_dataset,
    ]
)
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=st_eve_collate_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16'
)

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
# custom_hooks = [
#     dict(
#         type=EvaluateChatHook_ST,
#         tokenizer=tokenizer,
#         every_n_iters=evaluation_freq,
#         evaluation_inputs=evaluation_inputs,
#         evaluation_images=evaluation_images,
#         system='',)
# ]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=False,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)

init_cfg=None
