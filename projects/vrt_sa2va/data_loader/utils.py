import re
from typing import List
from pycocotools import mask as mask_utils
import numpy as np


def rle_to_mask(rle):
    """Convert RLE to mask"""
    mask = []
    for r in rle:
        m = mask_utils.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask


def extract_tagged_numbers_keep_original_tag(s: str) -> List[str]:
    """Extract object tags from tagged format <obj1>, <obj2>, etc. and keep the original tags
    Example:
        Input: "The cat is <obj1> and the dog is <obj2>"
        Output: ["<obj1>", "<obj2>"]
    """
    matches = re.findall(r"<obj\d+>", s)
    return matches


def extract_all_ver_tags(s: str) -> List[str]:    
    """Extract all ver tags from tagged format <ver>...</ver> and return the content inside the tags
    Example:
        Input: "The cat is <ver>chasing</ver> the dog"
        Output: ["chasing"]
    """
    matches = re.findall(r"<ver>(.*?)</ver>", s)
    return matches


def replace_ver_tags_with_nothing(s: str) -> str:
    """Replace <ver>...</ver> with nothing"""
    replaced_string = re.sub(r"<ver>.*?</ver>", "", s)
    return replaced_string


def extract_seg_token_return_psuedo_labels(s: str) -> List[str]:    
    """Extract segmentation tokens from tagged format <obj1>, <obj2>, etc. and return pseudo labels
    Example:
        Input: "The cat is <obj1> and the dog is <obj2>"
        Output: ["cat", "dog"]
    """
    matches = re.findall(r"\[SEG\]", s)
    return [f"object_{num}" for num in matches]



def replace_tagged_objects_with_special_token(s: str, special_token: str) -> str:
    """Replace <obj1>, <obj2>, etc. with special token"""
    def replace(match):
        return f"{match.group()} {special_token}"

    replaced_string = re.sub(r"<obj\d+>", replace, s)
    return replaced_string

