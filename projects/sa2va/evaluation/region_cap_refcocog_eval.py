import argparse
import re
import math
import os
import torch
import tqdm
from pycocotools import mask as _mask
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
import json

from utils import _init_dist_pytorch, get_dist_info, get_rank, collect_results_cpu

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocog region caption')
    parser.add_argument('model_path', help='hf model path.')
    parser.add_argument(
        '--output-path',
        default='./region_cap_pred.json',
        help='save path of the prediction')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    parser.add_argument('--data_root', default='./data', help='Root directory for all datasets.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

'''
class RegionCapInferenceDataset:
    def __init__(self,
                 image_folder,
                 annotation_file=None,
                 ):
        self.image_folder = image_folder
        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())

    def __len__(self):
        return len(self.image_dict_keys)

    def decode_mask(self, annotation, image_info):
        flag = False
        masks = []

        for ann_id in range(1):

            ann = {"segmentation": annotation}

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = _mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = _mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
        masks = np.stack(masks, axis=0)

        return masks

    def get_questions(self):
        # question = "<image>\nPlease give me a short description of the region in the picture marked by region1. Please response in a word."
        question = "<image>\nPlease give me a short description of the region in the picture marked by region1."
        return question

    def __getitem__(self, index):

        data_dict = {}

        image_id = self.image_dict_keys[index]
        image_file = self.image_dict[image_id]['file_name']

        questions = self.get_questions()

        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        image = Image.open(image_file).convert('RGB')

        masks = self.ann_dict[image_id]['segmentation']
        image_info = self.image_dict[image_id]
        masks = self.decode_mask(masks, image_info)

        data_dict['image'] = image
        data_dict['text'] = questions
        data_dict['img_id'] = image_id
        data_dict['mask_prompts'] = [masks]

        return data_dict
'''

class RegionCapInferenceDataset:
    def __init__(self,
                 image_folder,
                 annotation_file=None,
                 ):
        """
        Initializes the dataset.
        The key change is to directly access the dataset structure from the COCO object,
        as the default .imgs and .anns properties might be misleading for this custom format.
        """
        self.image_folder = image_folder
        print("Loading COCO annotation file for dataset...")
        self.coco = COCO(annotation_file)
        
        # The 'images' list in your file actually contains the caption data.
        # We will use this as our primary list of items to iterate over.
        self.caption_data_list = self.coco.dataset['images']
        
        # self.coco.anns is a dictionary mapping {annotation_id: annotation_object}.
        # This is exactly what we need to find the mask using the ID.
        self.annotations_dict = self.coco.anns
        print(f"Dataset initialized. Found {len(self.caption_data_list)} caption entries.")

    def __len__(self):
        """Returns the total number of caption entries."""
        return len(self.caption_data_list)

    def decode_mask(self, annotation_segmentation, image_info):
        """
        Decodes a segmentation mask from COCO's annotation format.
        This function remains largely the same as it correctly processes a segmentation entry.
        """
        masks = []
        
        # The input is the 'segmentation' part of an annotation object.
        ann = {"segmentation": annotation_segmentation}

        if not ann["segmentation"]: # Check for empty list
            m = np.zeros((image_info["height"], image_info["width"]), dtype=np.uint8)
            masks.append(m)
        else:
            if isinstance(ann["segmentation"][0], list):  # Polygon format
                rle = _mask.frPyObjects(ann["segmentation"], image_info["height"], image_info["width"])
            else:  # RLE format
                rle = ann["segmentation"]
            
            # Ensure RLE counts are bytes, as expected by pycocotools
            if isinstance(rle, list):
                for i in range(len(rle)):
                    if "counts" in rle[i] and not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()

            m = _mask.decode(rle)
            if len(m.shape) > 2:
                m = np.sum(m, axis=2)  # Sum over channels if multiple RLEs are decoded
            
            m = m.astype(np.uint8)
            masks.append(m)
            
        return np.stack(masks, axis=0)

    def get_questions(self):
        """Returns the prompt for the model."""
        question = "<image>\nPlease give me a short description of the region in the picture marked by region1."
        return question

    def __getitem__(self, index):
        """
        Fetches a single data item, ensuring the caption, image, and MASK are correctly matched.
        """
        data_dict = {}

        # 1. Get the current caption item from our primary list.
        caption_item = self.caption_data_list[index]
        
        # 2. Extract the critical ID. This ID is the bridge between caption and mask.
        item_id = caption_item['id']

        # 3. Get image file and basic info directly from the caption item.
        image_file = caption_item['file_name']
        image_info = {
            'height': caption_item['height'],
            'width': caption_item['width']
        }

        # 4. Find the CORRECT annotation (mask) using the item_id.
        # We look up this ID in the dictionary of all annotations.
        mask_segmentation = [] # Default to an empty mask
        if item_id in self.annotations_dict:
            annotation_object = self.annotations_dict[item_id]
            mask_segmentation = annotation_object['segmentation']
        else:
            # This is a critical failure, means our assumption about IDs is wrong.
            print(f"\n\nFATAL WARNING: Could not find an annotation with ID '{item_id}' to retrieve the mask. "
                  f"The model will receive an empty mask for image '{image_file}'. "
                  f"Please verify the data format.\n")

        # 5. Prepare the final data dictionary for the model.
        questions = self.get_questions()
        
        full_image_path = os.path.join(self.image_folder, image_file)
        try:
            image = Image.open(full_image_path).convert('RGB')
        except FileNotFoundError:
            print(f"\n\nFATAL WARNING: Image file not found at '{full_image_path}'. "
                  f"Skipping this item by returning an empty dictionary might be necessary in a real scenario.\n")
            # For now, let's create a dummy image to avoid crashing the whole process.
            image = Image.new('RGB', (image_info['width'], image_info['height']), color = 'red')

        
        # Decode the segmentation to get the final mask array.
        masks = self.decode_mask(mask_segmentation, image_info)

        data_dict['image'] = image
        data_dict['text'] = questions
        data_dict['img_id'] = item_id  # Pass the per-caption ID to be saved in the results file.
        data_dict['image_file'] = image_file
        data_dict['mask_prompts'] = [masks] # Model expects a list of masks

        return data_dict


ANNOTATION_FILE = './data/region_caption/refcocog/finetune_refcocog_val_with_mask.json'
IMAGE_FOLDER = './data/glamm_data/images/coco2014/train2014/'

def main():
    args = parse_args()

    annotation_file = os.path.join(args.data_root, 'region_caption/refcocog/finetune_refcocog_val_with_mask.json')
    image_folder = os.path.join(args.data_root, 'glamm_data/images/coco2014/train2014/')

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    dataset = RegionCapInferenceDataset(
        image_folder=image_folder,
        annotation_file=annotation_file,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        result_dict = {'image_id': data_batch['img_id'], 'image_file': data_batch['image_file']}
        del data_batch['img_id'], data_batch['image_file']

        prediction = model.predict_forward(**data_batch, tokenizer=tokenizer)['prediction']

        text_output = prediction.replace("<s>", "").replace("\n", "") \
            .replace("region1", '').replace("Region1", '').replace("The region marked by", "").replace("The region marked as", "").replace("The region marked", "") \
            .replace("is", "").replace("shows", "").replace(':', '').replace("   ", " ").replace("  ", " ")
        text_output = text_output.split("ASSISTANT: ")[-1]
        cleaned_str = re.sub(r'<.*?>', '', text_output)
        cleaned_str = cleaned_str.replace('[SEG]', '')
        cleaned_str = ' '.join(cleaned_str.split()).strip("'")
        cleaned_str = cleaned_str.strip()

        result_dict["caption"] = cleaned_str
        result_dict["prediction"] = cleaned_str
        results.append(result_dict)

    tmpdir = './dist_test_temp_regioncap_' + args.model_path.replace('/', '').replace('.', '')
    results = collect_results_cpu(results, len(dataset), tmpdir=tmpdir)
    if get_rank() == 0:
        with open(args.output_path, 'w') as json_file:
            json.dump(results, json_file, indent=2)

if __name__ == '__main__':
    main()
