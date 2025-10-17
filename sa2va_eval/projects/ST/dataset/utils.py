import torch
import torchvision.transforms as transforms
from math import ceil
from PIL import Image
import math
import numpy as np
from scipy.ndimage import label
from scipy.ndimage import distance_transform_edt

MAX_RESOLUTION = 1024 # 32 * 32

def get_resize_output_image_size(
    image_size,
    fix_resolution=False,
    max_resolution: int = MAX_RESOLUTION,
    patch_size=32
) -> tuple:
    if fix_resolution==True:
        return 224,224
    l1, l2 = image_size # 540, 32
    short, long = (l2, l1) if l2 <= l1 else (l1, l2)

    # set the nearest multiple of PATCH_SIZE for `long`
    requested_new_long = min(
        [
            ceil(long / patch_size) * patch_size,
            max_resolution,
        ]
    )

    new_long, new_short = requested_new_long, int(requested_new_long * short / long)
    # Find the nearest multiple of 64 for new_short
    new_short = ceil(new_short / patch_size) * patch_size
    return (new_long, new_short) if l2 <= l1 else (new_short, new_long)

def get_resize_output_image_size_new(
    image_size,
    fix_resolution=False,
    max_resolution: int = MAX_RESOLUTION,
    patch_size:int = None,
) -> tuple:
    if fix_resolution==True:
        return 384,384
    l1, l2 = image_size # 540, 32
    short, long = (l2, l1) if l2 <= l1 else (l1, l2)
    requested_new_long = ceil(long / patch_size) * patch_size
    new_long, new_short = requested_new_long, int(requested_new_long * short / long)
    # Find the nearest multiple of 64 for new_short
    new_short = ceil(new_short / patch_size) * patch_size
    token_num = new_long * new_short / (patch_size * patch_size)
    if token_num > 3600:
        scale_factor =  math.sqrt(token_num / 3600)
        new_long = int(new_long / scale_factor / patch_size) * patch_size
        new_short = int(new_short / scale_factor/ patch_size) * patch_size
    return (new_long, new_short) if l2 <= l1 else (new_short, new_long)


def preprocess_image(
    image_tensor: torch.Tensor,
    patch_size=32
) -> torch.Tensor:
    # Reshape the image to get the patches
    # shape changes: (C=3, H, W)
    # -> (C, N_H_PATCHES, W, PATCH_H)
    # -> (C, N_H_PATCHES, N_W_PATCHES, PATCH_H, PATCH_W)
    patches = image_tensor.unfold(1, patch_size, patch_size)\
        .unfold(2, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous() # -> (N_H_PATCHES, N_W_PATCHES, C, PATCH_H, PATCH_W)
    return patches

# def get_transform(height, width):
#     preprocess_transform = transforms.Compose([
#             transforms.Resize((height, width)),
#             transforms.ToTensor(),  # Convert the image to a PyTorch tensor
#             transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # Normalize with mean and
#                                 std=[0.26862954, 0.26130258, 0.27577711])   # standard deviation for pre-trained models on ImageNet
#         ])
#     return preprocess_transform

def get_transform(height, width):
    preprocess_transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with mean and
                                std=[0.229, 0.224, 0.225])   # standard deviation for pre-trained models on ImageNet
        ])
    return preprocess_transform

def get_transform_mask(height, width):
    preprocess_transform = transforms.Compose([
            transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        ])
    return preprocess_transform

def convert_image_to_patches(image, patch_size=32) -> torch.Tensor:
    # resize the image to the nearest multiple of 32
    width, height = image.size
    new_width, new_height = get_resize_output_image_size((width, height), patch_size=patch_size, fix_resolution=False)
    img_tensor = get_transform(new_height, new_width)(image) # 3， height, width
    # transform the process img to seq_length, 64*64*3
    img_patches = preprocess_image(img_tensor, patch_size=patch_size) # seq_length, 64*64*3
    return img_patches

def convert_mask_to_patches(mask, patch_size=32) -> torch.Tensor:
    mask = np.stack([mask, mask, mask], axis=2).astype(np.uint8)
    mask = Image.fromarray(mask)
    # resize the image to the nearest multiple of 32
    width, height = mask.size
    new_width, new_height = get_resize_output_image_size((width, height), patch_size=patch_size, fix_resolution=False)
    img_tensor = get_transform_mask(new_height, new_width)(mask) # 3， height, width
    # transform the process img to seq_length, 64*64*3
    img_patches = preprocess_image(img_tensor, patch_size=patch_size).flatten(2) # seq_length, 64*64*3
    img_patches = torch.sum(img_patches, dim=2) > 0
    return img_patches # (N_H_PATCHES, N_W_PATCHES, )

def has_multiple_connected_components(mask):
    labeled_mask, num_features = label(mask)
    return num_features > 1

def maskPrompt2boxPrompt(mask):
    # mask np.array bool
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        raise NotImplementedError
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    # print(x_min, " ", x_max, " ", y_min, " ", y_max)

    mask[y_min:y_max + 1, x_min:x_max + 1] = 1
    return mask

def maskPrompt2pointPrompt(mask, min_distance=16):
    distance_map = distance_transform_edt(mask)
    candidate_points = np.column_stack(np.where(distance_map > min_distance))

    if len(candidate_points) > 0:
        y, x = candidate_points[np.random.choice(len(candidate_points))]
    else:
        y, x = np.unravel_index(np.argmax(distance_map), distance_map.shape)
    result_mask = np.zeros_like(mask)
    result_mask[y, x] = 1
    return result_mask