from typing import List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np

from projects.vrt_sa2va.data_loader import extract_tagged_numbers_keep_original_tag, extract_seg_token_return_psuedo_labels


def simple_visualization(
        image, 
        masks: Optional[Union[np.ndarray, List[np.ndarray]]],
        prediction_text, 
        save_path=None
    ) -> None:
    """Create a multi-row visualization with at most 5 images per row"""
    if masks is None or len(masks) == 0:
        # If no masks, just show the original image
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.imshow(image)
        ax.set_title("Original Image (No masks generated)")
        ax.axis('off')
        plt.suptitle(f"Prediction: {prediction_text[:100]}...", fontsize=10)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        return

    # Calculate grid dimensions (max 5 images per row)
    total_images = len(masks) + 1  # +1 for original image
    max_images_per_row = 5
    n_cols = min(total_images, max_images_per_row)
    n_rows = (total_images + max_images_per_row - 1) // max_images_per_row  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # Ensure axes is always a flat array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1:
        # When single row with multiple columns, axes is already a 1D array
        axes = list(axes)
    else:
        # Multiple rows: flatten the 2D array
        axes = axes.flatten()
    
    # Show original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Show masks
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]  # RGB tuples
    tags = extract_tagged_numbers_keep_original_tag(prediction_text)
    if len(masks) != len(tags) and len(tags) == 0:
        tags = extract_seg_token_return_psuedo_labels(prediction_text)
    
    # assert len(masks) == len(tags), "Number of masks must match number of tags in prediction text"
    if len(masks) != len(tags):
        print("Warning: Number of masks does not match number of tags in prediction text. Using only available masks.")
    if len(masks) > len(tags):
        masks = masks[:len(tags)]
    if len(tags) > len(masks):
        tags = tags[:len(masks)]
    
    for i, mask in enumerate(masks):
        ax_idx = i + 1
        ax = axes[ax_idx]
        
        # Create overlay
        img_array = np.array(image)
        mask_overlay = np.zeros_like(img_array)
        color_rgb = colors[i % len(colors)]
        
        for c in range(3):
            mask_overlay[:, :, c] = mask * color_rgb[c] * 255
        
        # Blend image and mask
        alpha = 0.6
        blended = img_array * (1 - alpha) + mask_overlay * alpha
        
        ax.imshow(blended.astype(np.uint8))
        ax.set_title(tags[i])
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(total_images, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f"Prediction: {prediction_text[:100]}...", fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
