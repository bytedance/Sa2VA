from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mask_utils
import numpy as np


COLORS = [
    (255, 69, 58), # Scarlet
    (65, 105, 225), # Royal Blue
    (255, 191, 0), # Amber
    (60, 179, 113), # Grass Green
    (138, 43, 226), # Violet
    (0, 188, 212), # Cyan
    (255, 105, 180), # Sakura Pink
    (54, 69, 79), # Charcoal
    (255, 215, 0), # Gold
    (191, 255, 0), # Lime
]


def visualize_mask(image: Image.Image, pred_mask: Image.Image) -> Image.Image:
    """
    Visualize the predicted mask on the image.
    Args:
        image (PIL.Image): The input image. (RGB format)
        pred_mask (PIL.Image): The predicted mask to overlay on the image. (L format)
    Returns:
        PIL.Image: The image with the predicted mask overlay.
    """

    result = image.convert("RGBA")
    overlay = Image.new("RGBA", result.size, (255, 0, 0, 200))
    result = Image.alpha_composite(result, Image.composite(overlay, result, pred_mask))
    result = result.convert("RGB")

    return result



def visualize_n_masks(image: Image.Image, pred_masks: list[Image.Image]) -> Image.Image:
    """
    Visualize the predicted mask on the image.
    Args:
        image (PIL.Image): The input image. (RGB format)
        pred_masks List of (PIL.Image): The predicted masks to overlay on the image. (L format)
    Returns:
        PIL.Image: The image with the predicted mask overlay.
    """
    result = image.convert("RGBA")

    for idx, mask in enumerate(pred_masks):
        color_idx = idx % len(COLORS)
        color_tuple = COLORS[color_idx] + (200,)
        overlay = Image.new("RGBA", result.size, color_tuple)
        result = Image.alpha_composite(result, Image.composite(overlay, result, mask.convert("L")))

    return result.convert("RGB")

def rle_to_mask(rle):
    mask = []
    for r in rle:
        m = mask_utils.decode(r)
        m = np.uint8(m)
        mask.append(m)
    mask = np.stack(mask, axis=0)
    return mask

