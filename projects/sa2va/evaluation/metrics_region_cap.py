import argparse
import os
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

def parse_args():
    parser = argparse.ArgumentParser(description="GLaMM Inference - Region Captioning")
    parser.add_argument("--annotation_file",
                        default=None, type=str,
                        help="Path to the custom COCO-style ground truth annotation file.")
    parser.add_argument("--results_dir", default="results", type=str, help="The path to the prediction results file.")
    parser.add_argument("--data_root", default="./data", help="Root directory for all datasets.")
    return parser.parse_args()

def convert_custom_format_to_coco_caption(original_data):
    """
    Converts the custom data format to a standard COCO Caption format in memory.
    It extracts unique images and captions from the 'images' list of the original data,
    and completely rebuilds the 'images' and 'annotations' lists for evaluation.
    """
    print("Converting custom data format to standard COCO Caption format...")
    source_items = original_data.get('images', [])
    if not source_items:
        print("Error: 'images' list is empty or not found in the source data.")
        return None
    new_images = []
    new_annotations = []
    unique_image_tracker = {}
    new_image_id_counter = 1
    new_annotation_id_counter = 1
    for item in source_items:
        original_id = item.get('original_id')
        if original_id is None:
            print(f"Warning: Skipping item without 'original_id': {item}")
            continue
        if original_id not in unique_image_tracker:
            new_id = new_image_id_counter
            unique_image_tracker[original_id] = new_id
            # *** MODIFICATION START ***
            # Add 'original_id' to the new image object so we can use it for mapping later.
            image_obj = {
                'id': new_id,
                'original_id': original_id, # <--- THE FIX IS HERE!
                'file_name': item.get('file_name'),
                'height': item.get('height'),
                'width': item.get('width')
            }
            # *** MODIFICATION END ***
            new_images.append(image_obj)
            new_image_id_counter += 1
        else:
            new_id = unique_image_tracker[original_id]
        caption = item.get('caption')
        if caption:
            ann_obj = {
                'id': new_annotation_id_counter,
                'image_id': new_id,
                'caption': caption
            }
            new_annotations.append(ann_obj)
            new_annotation_id_counter += 1
    coco_caption_data = {
        'info': original_data.get('info', {}),
        'licenses': original_data.get('licenses', []),
        'images': new_images,
        'annotations': new_annotations
    }
    print(f"Conversion complete. Created {len(new_images)} unique images and {len(new_annotations)} annotations.")
    return coco_caption_data

def main():
    args = parse_args()

    annotation_file = args.annotation_file
    if annotation_file is None:
        annotation_file = os.path.join(args.data_root, 'region_caption/refcocog/finetune_refcocog_val_with_mask.json')

    # 1. Load the original, custom-formatted JSON file
    print(f"Loading original annotation file: {annotation_file}")
    with open(annotation_file, 'r') as f:
        original_gt_data = json.load(f)

    # 2. Convert the loaded data into a standard format in memory
    standard_gt_data = convert_custom_format_to_coco_caption(original_gt_data)
    
    if standard_gt_data is None:
        print("Error during data conversion. Exiting.")
        return

    # 3. Initialize COCO with the newly created standard dictionary
    coco = COCO()
    coco.dataset = standard_gt_data
    coco.createIndex()

    # 4. Load prediction results.
    # IMPORTANT: The prediction file must also be adjusted to use the new image IDs.
    # We will load it and map the IDs.
    print(f"Loading prediction file: {args.results_dir}")
    with open(args.results_dir, 'r') as f:
        predictions = json.load(f)

    # Create a map from original_id to new_id for predictions
    # This assumes predictions use the 'image_id' from the original 'images' list (e.g., 347813)
    # Let's refine this. The prediction script likely outputs based on the unique image, not the caption.
    # Let's assume the prediction file's 'image_id' corresponds to our 'original_id'.
    
    # We need a map from the original file's unique image ID to our new unique image ID.
    # Let's rebuild the `unique_image_tracker` map from `original_id` to `new_id`.
    unique_image_tracker = {img['original_id']: img['id'] for img in standard_gt_data['images']}

    # Now, update the image_id in the predictions
    updated_predictions = []
    for pred in predictions:
        # We need to know what ID the prediction file uses. Let's assume it's 'image_id'
        # and that it corresponds to the 'original_id' in the GT file.
        original_pred_id = pred.get('image_id') 
        if original_pred_id in unique_image_tracker:
            pred['image_id'] = unique_image_tracker[original_pred_id]
            updated_predictions.append(pred)
        else:
            # This part is tricky. If the prediction ID doesn't match, we need a better mapping strategy.
            # For now, let's assume a direct mapping works.
            pass
    
    if not updated_predictions:
         print("Warning: Could not map any prediction IDs to the new ground truth IDs. Using original predictions.")
         updated_predictions = predictions


    # Load the (potentially updated) prediction results into COCO
    coco_result = coco.loadRes(updated_predictions)

    # Create and run evaluation
    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.params['image_id'] = coco_result.getImgIds()
    coco_eval.evaluate()

    # Print and save results
    # ... (rest of the code is the same)
    output_file_path = f"./region_cap_metrics.txt"
    print("\n--- Evaluation Metrics ---")
    with open(output_file_path, 'w') as f:
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')
            f.write(f"{metric}: {score:.3f}\n")
    print(f"\nMetrics saved to {output_file_path}")


if __name__ == "__main__":
    main()
