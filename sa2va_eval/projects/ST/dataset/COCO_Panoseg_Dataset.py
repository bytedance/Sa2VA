import copy
import json
import random
import os
import torch
from mmengine import print_log
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
from xtuner.registry import BUILDER
from xtuner.utils import IGNORE_INDEX
from .utils import convert_image_to_patches
from torch.utils.data import Dataset
from panopticapi.utils import rgb2id

SEG_QUESTIONS = [
    "Can you segment the {class_name} in this image?",
    "Please segment {class_name} in this image.",
    "What is {class_name} in this image? Please respond with segmentation mask.",
    "What is {class_name} in this image? Please output segmentation mask.",

    "Can you segment the {class_name} in this image",
    "Please segment {class_name} in this image",
    "What is {class_name} in this image? Please respond with segmentation mask",
    "What is {class_name} in this image? Please output segmentation mask",

    "Could you provide a segmentation mask for the {class_name} in this image?",
    "Please identify and segment the {class_name} in this image.",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask.",
    "Can you highlight the {class_name} in this image with a segmentation mask?",

    "Could you provide a segmentation mask for the {class_name} in this image",
    "Please identify and segment the {class_name} in this image",
    "Where is the {class_name} in this picture? Please respond with a segmentation mask",
    "Can you highlight the {class_name} in this image with a segmentation mask",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

COCO_CATEGORIES = [
    {"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "person"},
    {"color": [119, 11, 32], "isthing": 1, "id": 2, "name": "bicycle"},
    {"color": [0, 0, 142], "isthing": 1, "id": 3, "name": "car"},
    {"color": [0, 0, 230], "isthing": 1, "id": 4, "name": "motorcycle"},
    {"color": [106, 0, 228], "isthing": 1, "id": 5, "name": "airplane"},
    {"color": [0, 60, 100], "isthing": 1, "id": 6, "name": "bus"},
    {"color": [0, 80, 100], "isthing": 1, "id": 7, "name": "train"},
    {"color": [0, 0, 70], "isthing": 1, "id": 8, "name": "truck"},
    {"color": [0, 0, 192], "isthing": 1, "id": 9, "name": "boat"},
    {"color": [250, 170, 30], "isthing": 1, "id": 10, "name": "traffic light"},
    {"color": [100, 170, 30], "isthing": 1, "id": 11, "name": "fire hydrant"},
    {"color": [220, 220, 0], "isthing": 1, "id": 13, "name": "stop sign"},
    {"color": [175, 116, 175], "isthing": 1, "id": 14, "name": "parking meter"},
    {"color": [250, 0, 30], "isthing": 1, "id": 15, "name": "bench"},
    {"color": [165, 42, 42], "isthing": 1, "id": 16, "name": "bird"},
    {"color": [255, 77, 255], "isthing": 1, "id": 17, "name": "cat"},
    {"color": [0, 226, 252], "isthing": 1, "id": 18, "name": "dog"},
    {"color": [182, 182, 255], "isthing": 1, "id": 19, "name": "horse"},
    {"color": [0, 82, 0], "isthing": 1, "id": 20, "name": "sheep"},
    {"color": [120, 166, 157], "isthing": 1, "id": 21, "name": "cow"},
    {"color": [110, 76, 0], "isthing": 1, "id": 22, "name": "elephant"},
    {"color": [174, 57, 255], "isthing": 1, "id": 23, "name": "bear"},
    {"color": [199, 100, 0], "isthing": 1, "id": 24, "name": "zebra"},
    {"color": [72, 0, 118], "isthing": 1, "id": 25, "name": "giraffe"},
    {"color": [255, 179, 240], "isthing": 1, "id": 27, "name": "backpack"},
    {"color": [0, 125, 92], "isthing": 1, "id": 28, "name": "umbrella"},
    {"color": [209, 0, 151], "isthing": 1, "id": 31, "name": "handbag"},
    {"color": [188, 208, 182], "isthing": 1, "id": 32, "name": "tie"},
    {"color": [0, 220, 176], "isthing": 1, "id": 33, "name": "suitcase"},
    {"color": [255, 99, 164], "isthing": 1, "id": 34, "name": "frisbee"},
    {"color": [92, 0, 73], "isthing": 1, "id": 35, "name": "skis"},
    {"color": [133, 129, 255], "isthing": 1, "id": 36, "name": "snowboard"},
    {"color": [78, 180, 255], "isthing": 1, "id": 37, "name": "sports ball"},
    {"color": [0, 228, 0], "isthing": 1, "id": 38, "name": "kite"},
    {"color": [174, 255, 243], "isthing": 1, "id": 39, "name": "baseball bat"},
    {"color": [45, 89, 255], "isthing": 1, "id": 40, "name": "baseball glove"},
    {"color": [134, 134, 103], "isthing": 1, "id": 41, "name": "skateboard"},
    {"color": [145, 148, 174], "isthing": 1, "id": 42, "name": "surfboard"},
    {"color": [255, 208, 186], "isthing": 1, "id": 43, "name": "tennis racket"},
    {"color": [197, 226, 255], "isthing": 1, "id": 44, "name": "bottle"},
    {"color": [171, 134, 1], "isthing": 1, "id": 46, "name": "wine glass"},
    {"color": [109, 63, 54], "isthing": 1, "id": 47, "name": "cup"},
    {"color": [207, 138, 255], "isthing": 1, "id": 48, "name": "fork"},
    {"color": [151, 0, 95], "isthing": 1, "id": 49, "name": "knife"},
    {"color": [9, 80, 61], "isthing": 1, "id": 50, "name": "spoon"},
    {"color": [84, 105, 51], "isthing": 1, "id": 51, "name": "bowl"},
    {"color": [74, 65, 105], "isthing": 1, "id": 52, "name": "banana"},
    {"color": [166, 196, 102], "isthing": 1, "id": 53, "name": "apple"},
    {"color": [208, 195, 210], "isthing": 1, "id": 54, "name": "sandwich"},
    {"color": [255, 109, 65], "isthing": 1, "id": 55, "name": "orange"},
    {"color": [0, 143, 149], "isthing": 1, "id": 56, "name": "broccoli"},
    {"color": [179, 0, 194], "isthing": 1, "id": 57, "name": "carrot"},
    {"color": [209, 99, 106], "isthing": 1, "id": 58, "name": "hot dog"},
    {"color": [5, 121, 0], "isthing": 1, "id": 59, "name": "pizza"},
    {"color": [227, 255, 205], "isthing": 1, "id": 60, "name": "donut"},
    {"color": [147, 186, 208], "isthing": 1, "id": 61, "name": "cake"},
    {"color": [153, 69, 1], "isthing": 1, "id": 62, "name": "chair"},
    {"color": [3, 95, 161], "isthing": 1, "id": 63, "name": "couch"},
    {"color": [163, 255, 0], "isthing": 1, "id": 64, "name": "potted plant"},
    {"color": [119, 0, 170], "isthing": 1, "id": 65, "name": "bed"},
    {"color": [0, 182, 199], "isthing": 1, "id": 67, "name": "dining table"},
    {"color": [0, 165, 120], "isthing": 1, "id": 70, "name": "toilet"},
    {"color": [183, 130, 88], "isthing": 1, "id": 72, "name": "tv"},
    {"color": [95, 32, 0], "isthing": 1, "id": 73, "name": "laptop"},
    {"color": [130, 114, 135], "isthing": 1, "id": 74, "name": "mouse"},
    {"color": [110, 129, 133], "isthing": 1, "id": 75, "name": "remote"},
    {"color": [166, 74, 118], "isthing": 1, "id": 76, "name": "keyboard"},
    {"color": [219, 142, 185], "isthing": 1, "id": 77, "name": "cell phone"},
    {"color": [79, 210, 114], "isthing": 1, "id": 78, "name": "microwave"},
    {"color": [178, 90, 62], "isthing": 1, "id": 79, "name": "oven"},
    {"color": [65, 70, 15], "isthing": 1, "id": 80, "name": "toaster"},
    {"color": [127, 167, 115], "isthing": 1, "id": 81, "name": "sink"},
    {"color": [59, 105, 106], "isthing": 1, "id": 82, "name": "refrigerator"},
    {"color": [142, 108, 45], "isthing": 1, "id": 84, "name": "book"},
    {"color": [196, 172, 0], "isthing": 1, "id": 85, "name": "clock"},
    {"color": [95, 54, 80], "isthing": 1, "id": 86, "name": "vase"},
    {"color": [128, 76, 255], "isthing": 1, "id": 87, "name": "scissors"},
    {"color": [201, 57, 1], "isthing": 1, "id": 88, "name": "teddy bear"},
    {"color": [246, 0, 122], "isthing": 1, "id": 89, "name": "hair drier"},
    {"color": [191, 162, 208], "isthing": 1, "id": 90, "name": "toothbrush"},
    {"color": [255, 255, 128], "isthing": 0, "id": 92, "name": "banner"},
    {"color": [147, 211, 203], "isthing": 0, "id": 93, "name": "blanket"},
    {"color": [150, 100, 100], "isthing": 0, "id": 95, "name": "bridge"},
    {"color": [168, 171, 172], "isthing": 0, "id": 100, "name": "cardboard"},
    {"color": [146, 112, 198], "isthing": 0, "id": 107, "name": "counter"},
    {"color": [210, 170, 100], "isthing": 0, "id": 109, "name": "curtain"},
    {"color": [92, 136, 89], "isthing": 0, "id": 112, "name": "door-stuff"},
    {"color": [218, 88, 184], "isthing": 0, "id": 118, "name": "floor-wood"},
    {"color": [241, 129, 0], "isthing": 0, "id": 119, "name": "flower"},
    {"color": [217, 17, 255], "isthing": 0, "id": 122, "name": "fruit"},
    {"color": [124, 74, 181], "isthing": 0, "id": 125, "name": "gravel"},
    {"color": [70, 70, 70], "isthing": 0, "id": 128, "name": "house"},
    {"color": [255, 228, 255], "isthing": 0, "id": 130, "name": "light"},
    {"color": [154, 208, 0], "isthing": 0, "id": 133, "name": "mirror-stuff"},
    {"color": [193, 0, 92], "isthing": 0, "id": 138, "name": "net"},
    {"color": [76, 91, 113], "isthing": 0, "id": 141, "name": "pillow"},
    {"color": [255, 180, 195], "isthing": 0, "id": 144, "name": "platform"},
    {"color": [106, 154, 176], "isthing": 0, "id": 145, "name": "playingfield"},
    {"color": [230, 150, 140], "isthing": 0, "id": 147, "name": "railroad"},
    {"color": [60, 143, 255], "isthing": 0, "id": 148, "name": "river"},
    {"color": [128, 64, 128], "isthing": 0, "id": 149, "name": "road"},
    {"color": [92, 82, 55], "isthing": 0, "id": 151, "name": "roof"},
    {"color": [254, 212, 124], "isthing": 0, "id": 154, "name": "sand"},
    {"color": [73, 77, 174], "isthing": 0, "id": 155, "name": "sea"},
    {"color": [255, 160, 98], "isthing": 0, "id": 156, "name": "shelf"},
    {"color": [255, 255, 255], "isthing": 0, "id": 159, "name": "snow"},
    {"color": [104, 84, 109], "isthing": 0, "id": 161, "name": "stairs"},
    {"color": [169, 164, 131], "isthing": 0, "id": 166, "name": "tent"},
    {"color": [225, 199, 255], "isthing": 0, "id": 168, "name": "towel"},
    {"color": [137, 54, 74], "isthing": 0, "id": 171, "name": "wall-brick"},
    {"color": [135, 158, 223], "isthing": 0, "id": 175, "name": "wall-stone"},
    {"color": [7, 246, 231], "isthing": 0, "id": 176, "name": "wall-tile"},
    {"color": [107, 255, 200], "isthing": 0, "id": 177, "name": "wall-wood"},
    {"color": [58, 41, 149], "isthing": 0, "id": 178, "name": "water-other"},
    {"color": [183, 121, 142], "isthing": 0, "id": 180, "name": "window-blind"},
    {"color": [255, 73, 97], "isthing": 0, "id": 181, "name": "window-other"},
    {"color": [107, 142, 35], "isthing": 0, "id": 184, "name": "tree-merged"},
    {"color": [190, 153, 153], "isthing": 0, "id": 185, "name": "fence-merged"},
    {"color": [146, 139, 141], "isthing": 0, "id": 186, "name": "ceiling-merged"},
    {"color": [70, 130, 180], "isthing": 0, "id": 187, "name": "sky-other-merged"},
    {"color": [134, 199, 156], "isthing": 0, "id": 188, "name": "cabinet-merged"},
    {"color": [209, 226, 140], "isthing": 0, "id": 189, "name": "table-merged"},
    {"color": [96, 36, 108], "isthing": 0, "id": 190, "name": "floor-other-merged"},
    {"color": [96, 96, 96], "isthing": 0, "id": 191, "name": "pavement-merged"},
    {"color": [64, 170, 64], "isthing": 0, "id": 192, "name": "mountain-merged"},
    {"color": [152, 251, 152], "isthing": 0, "id": 193, "name": "grass-merged"},
    {"color": [208, 229, 228], "isthing": 0, "id": 194, "name": "dirt-merged"},
    {"color": [206, 186, 171], "isthing": 0, "id": 195, "name": "paper-merged"},
    {"color": [152, 161, 64], "isthing": 0, "id": 196, "name": "food-other-merged"},
    {"color": [116, 112, 0], "isthing": 0, "id": 197, "name": "building-other-merged"},
    {"color": [0, 114, 143], "isthing": 0, "id": 198, "name": "rock-merged"},
    {"color": [102, 102, 156], "isthing": 0, "id": 199, "name": "wall-other-merged"},
    {"color": [250, 141, 255], "isthing": 0, "id": 200, "name": "rug-merged"},
]

def read_class_texts(path):
    with open(path, 'r') as f:
        texts = f.readlines()

    id2text_category = {}
    for text in texts:
        text = text.replace("\n", "").strip()
        _id, categories = text.split(":")
        _id = int(_id)
        category = categories.split(",")[0].strip()
        if category == "invalid_class_id":
            category = None
        id2text_category[_id] = category
    return id2text_category

NON_VISION_TOKEN = -1
PROMPT_TMPL = '<|im_start|>user\n{input}<|im_end|>\n'

class COCOSegmDataset(Dataset):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    IMG_CONTEXT_TOKEN = "<vpatch>"
    IMG_START_TOKEN = "<vision>"
    IMG_END_TOKEN = "</vision>"

    IMG_RSEP_TOKEN = "<vrow_sep>"
    CLS_TOKEN = "<|vis_cls|>"

    def __init__(self,
                 image_folder,
                 pano_rgb_folder,
                 pano_json,
                 class_txt,
                 special_tokens=None,
                 prompt_template=None,
                 tokenizer=None,
                 max_length=2048,
                 num_classes_per_sample=3,
                 patch_size=32,
                 add_cls=False,
                 **kwargs):
        self._system = ''

        self.patch_size = patch_size
        self.add_cls = add_cls
        self.tokenizer = BUILDER.build(tokenizer)
        self.tokenizer.vis_beg_tok = "<vision>"
        self.tokenizer.vis_patch_tok = "<vpatch>"
        self.tokenizer.vis_rsep_tok = "<vrow_sep>"
        self.tokenizer.vis_frm_tok = "<vframe_sep>"
        self.tokenizer.vis_end_tok = "</vision>"
        self.tokenizer.vis_cls_tok = "<|vis_cls|>"
        if special_tokens is not None:
            self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.tokenizer.vis_beg_tok_id = self.tokenizer.convert_tokens_to_ids("<vision>")
        self.tokenizer.vis_patch_tok_id = self.tokenizer.convert_tokens_to_ids("<vpatch>")
        self.tokenizer.vis_rsep_tok_id = self.tokenizer.convert_tokens_to_ids("<vrow_sep>")
        self.tokenizer.vis_frm_tok_id = self.tokenizer.convert_tokens_to_ids("<vframe_sep>")
        self.tokenizer.vis_end_tok_id = self.tokenizer.convert_tokens_to_ids("</vision>")
        self.tokenizer.vis_cls_tok_id = self.tokenizer.convert_tokens_to_ids("<|vis_cls|>")

        self.image_folder = image_folder
        self.pano_rgb_folder = pano_rgb_folder
        self.pano_json = pano_json
        self.id2name = read_class_texts(class_txt)

        self.datas = self.read_pano_json()

        self.template = prompt_template
        self.template['INSTRUCTION'] = PROMPT_TMPL
        self.template['SUFFIX'] = '<|endoftext|>'
        self.max_length = max_length

        self.num_classes_per_sample = num_classes_per_sample
        self._max_refetch = 1000

        print("COCO panoptic dataset, include {} items.".format(len(self.datas)))

    @property
    def modality_length(self):
        length_list = []
        for idx in range(len(self.datas)):
            length_list.append(100)
        return length_list

    def __len__(self):
        return len(self.datas)

    def read_pano_json(self):

        id2meta_info = {}
        for meta_info in COCO_CATEGORIES:
            id2meta_info[meta_info["id"]] = meta_info

        with open(self.pano_json, 'r') as f:
            json_info = json.load(f)
        ret = []
        for ann in json_info["annotations"]:
            image_id = int(ann["image_id"])
            image_file = os.path.join(self.image_folder, os.path.splitext(ann["file_name"])[0] + ".jpg")
            label_file = os.path.join(self.pano_rgb_folder, ann["file_name"])
            segments_info = copy.deepcopy(ann["segments_info"])
            for _info in segments_info:
                _info.update({"isthing": id2meta_info[_info["category_id"]]["isthing"]}) # 0 is stuff, 1 is thing
            ret.append(
                {
                    "file_name": image_file,
                    "image_id": image_id,
                    "pan_seg_file_name": label_file,
                    "segments_info": segments_info,
                }
            )
        return ret

    def sort_masks(self, masks):
        ret = []
        sx = []
        for mask in masks:
            _y,_x = np.nonzero(mask)
            sx.append(np.min(_x))
        indexes = np.argsort(np.array(sx))
        for index in indexes:
            ret.append(masks[index])
        return ret

    def _parse_annotations(self, ann_info):
        image_path = ann_info['file_name']
        image = Image.open(image_path).convert('RGB')
        width, height = image.size

        pano_seg = Image.open(ann_info["pan_seg_file_name"]).convert('RGB')
        segment_infos = ann_info["segments_info"]
        pan_seg_gt = rgb2id(np.array(pano_seg))

        classes, masks, isthings = [], [], []
        for segment_info in segment_infos:
            class_id = int(segment_info["category_id"])
            if not segment_info["iscrowd"]:
                classes.append(class_id)
                masks.append(pan_seg_gt == segment_info["id"])
                isthings.append(segment_info["isthing"])

        valid_classes = list(set(classes))
        if len(valid_classes) > self.num_classes_per_sample:
            random.shuffle(valid_classes)
            valid_classes = valid_classes[:self.num_classes_per_sample]

        per_class_anns = {}
        for obj_class, mask, isthing in zip(classes, masks, isthings):
            if obj_class in valid_classes:
                if obj_class not in per_class_anns.keys():
                    per_class_anns[obj_class] = {"masks": [mask], "isthing": isthing, "text_class": self.id2name[obj_class]}
                else:
                    per_class_anns[obj_class]["masks"].append(mask)

        for class_name in per_class_anns.keys():
            per_class_anns[class_name]["masks"] = self.sort_masks(per_class_anns[class_name]["masks"])

        conversation = []
        ret_masks = []
        for i, class_id in enumerate(per_class_anns.keys()):
            masks = per_class_anns[class_id]["masks"]
            text_class = per_class_anns[class_id]["text_class"]
            isthing = per_class_anns[class_id]["isthing"]

            question = random.choice(SEG_QUESTIONS).format(class_name=text_class)
            if isthing == 1:
                question = question + " Segmentation in the instance mode."
                answer = ""
                for i_mask in range(len(masks)):
                    answer += f"{text_class}-{i_mask+1} [SEG] "
                answer = answer.strip() + "."
            else:
                question = question + " Segmentation in the semantic mode."
                answer = text_class + " [SEG]."
            if i == 0:
                question = question
            conversation.append({'from': 'human', 'value': question})
            conversation.append({'from': 'gpt', 'value': answer})
            ret_masks += masks
        # print(conversation)
        if len(ret_masks) == 0:
            return None
        masks = torch.stack([torch.from_numpy(mask) for mask in ret_masks], dim=0)

        ann_info.update({
            'masks': masks,
            'conversations': conversation,
            'image': image_path
        })
        return ann_info

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

    def prepare_data(self, index):
        data_dict = self.datas[index]
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {'vision_patch_idx': self.tokenizer.vis_patch_tok_id}
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            try:
                image = Image.open(image_file).convert('RGB')
                out_data_dict['image'] = image
            except Exception as e:
                print(f'Error: {e}', flush=True)
                print_log(f'Error: {e}', logger='current')
                return None

            image_patches = convert_image_to_patches(image, self.patch_size)
            # tensor, (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
            h_patches, w_patches = image_patches.shape[:2]
            out_data_dict['vision_patches'] = image_patches.flatten(0, 1).flatten(
                1)  # (n_patches, 3*patch_size*patch_size)
            out_data_dict['patch_nums_per_images'] = (h_patches, w_patches)

            image_token_str, image_token_len, image_token_patch_indices = \
                self.prepare_image_textual_seq_norowsep(
                    image_patches.shape[0], image_patches.shape[1]
                )

            token_dict = self.get_inputid_labels(
                data_dict['conversations'], image_token_str, image_token_patch_indices)
            out_data_dict.update(token_dict)

            out_data_dict.update(token_dict)
        else:
            out_data_dict['patch_nums_per_images'] = (0, 0)
            token_dict = self.get_inputid_labels(
                data_dict['conversations'], "", [])
            out_data_dict.update(token_dict)
        return out_data_dict

    def get_inputid_labels(self, conversations, image_token_str, image_token_patch_indices) -> dict:
        input = ''
        out_conversation = []
        while conversations and conversations[0]['from'] == 'gpt':
            # Skip the first one if it is from gpt
            conversations = conversations[1:]

        # remove image token from text conversation
        for i, msg in enumerate(conversations):
            if msg['from'] == 'human':
                # change to 1 image
                if '<image>' in msg['value']:
                    msg['value'] = msg['value'].replace('<image>\n', '').replace('\n<image>', '').replace('<image>', '')
                input += msg['value'].strip()
            elif msg['from'] == 'gpt':
                out_conversation.append({
                    'input': input,
                    'output': msg['value'].strip()
                })
                input = ''
            else:
                raise NotImplementedError

        input_ids, labels = [], []
        token_patch_indices = []

        # firstly add the images strs
        image_token_str_tokens = self.tokenizer.encode(image_token_str, add_special_tokens=False)
        input_ids += image_token_str_tokens
        labels += [IGNORE_INDEX] * len(image_token_str_tokens)
        token_patch_indices += image_token_patch_indices

        for i, single_turn_conversation in enumerate(out_conversation):
            input = single_turn_conversation.get('input', '')
            if input is None:
                input = ''
            input_text = self.template.INSTRUCTION.format(
                input=input, round=i + 1)

            if i == 0:
                if self._system != '' and self._system is not None:
                    system = self.template.SYSTEM.format(system=self._system)
                    input_text = system + input_text
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=True)
            else:
                input_encode = self.tokenizer.encode(
                    input_text, add_special_tokens=False)
            input_ids += input_encode
            labels += [IGNORE_INDEX] * len(input_encode)
            token_patch_indices += [NON_VISION_TOKEN] * len(input_encode)

            output_text = single_turn_conversation.get('output', '')
            if self.template.get('SUFFIX', None):
                output_text += self.template.SUFFIX
            output_encode = self.tokenizer.encode(
                output_text, add_special_tokens=False)
            input_ids += output_encode
            labels += copy.deepcopy(output_encode)
            token_patch_indices += [NON_VISION_TOKEN] * len(output_encode)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            labels = labels[:self.max_length]
            token_patch_indices = token_patch_indices[:self.max_length]
            print_log(
                f'Warning: input_ids length({len(input_ids)}) '
                f'is longer than max_length, cut to {self.max_length}',
                logger='current')
        vision_start_end = self.search_vision_tokens(input_ids)
        return {'input_ids': input_ids, 'labels': labels,
                'vision_patch_indices': token_patch_indices,
                'vision_start_end': vision_start_end,
                }

    def _rand_another(self):
        idx = random.randint(0, len(self.datas))
        return idx

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            data = self.prepare_data(index)
            # Broken images may cause the returned data to be None
            if data is None:
                index = self._rand_another()
                continue
            return data

    def search_vision_tokens(self, input_ids):
        image_start_idx = self.tokenizer(self.IMG_START_TOKEN, add_special_tokens=False).input_ids[0]
        image_end_idx = self.tokenizer(self.IMG_END_TOKEN, add_special_tokens=False).input_ids[0]
        if image_start_idx not in input_ids:
            return None
        else:
            start_idx = input_ids.index(image_start_idx)
            end_idx = input_ids.index(image_end_idx)
            return [start_idx + 1, end_idx]
