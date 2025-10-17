import os
import json
import argparse
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils
from pycocoevalcap.eval import COCOEvalCap
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--split", required=True, help="Evaluation split, options are 'val', 'test'")
    parser.add_argument("--prediction_dir_path", required=True, help="The path where the inference results are stored.")
    parser.add_argument("--gt_dir_path", required=False, default="./data/glamm_data/annotations/gcg_val_test/",
                        help="The path containing GranD-f evaluation annotations.")

    args = parser.parse_args()

    return args


# Load pre-trained model tokenizer and model for evaluation
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    # Use the mean of the last hidden states as sentence embedding
    sentence_embedding = torch.mean(outputs.last_hidden_state[0], dim=0).detach().numpy()

    return sentence_embedding

def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)

    return iou

def bbox_to_x1y1x2y2(bbox):
    x1, y1, w, h = bbox
    bbox = [x1, y1, x1 + w, y1 + h]

    return bbox

def compute_miou(pred_masks, gt_masks):
    # Computing mIoU between predicted masks and ground truth masks
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    # One-to-one pairing and mean IoU calculation
    paired_iou = []
    while iou_matrix.size > 0 and np.max(iou_matrix) > 0:
        max_iou_idx = np.unravel_index(np.argmax(iou_matrix, axis=None), iou_matrix.shape)
        paired_iou.append(iou_matrix[max_iou_idx])
        iou_matrix = np.delete(iou_matrix, max_iou_idx[0], axis=0)
        iou_matrix = np.delete(iou_matrix, max_iou_idx[1], axis=1)

    return np.mean(paired_iou) if paired_iou else 0.0


def evaluate_mask_miou(coco_gt, image_ids, pred_save_path):
    # Load predictions
    coco_dt = coco_gt.loadRes(pred_save_path)

    mious = []
    for image_id in tqdm(image_ids):
        # Getting ground truth masks
        matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == image_id]
        ann_ids = [ann['id'] for ann in matching_anns]

        gt_anns = coco_gt.loadAnns(ann_ids)
        gt_masks = [maskUtils.decode(ann['segmentation']) for ann in gt_anns if 'segmentation' in ann]

        # Getting predicted masks
        matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == image_id]
        dt_ann_ids = [ann['id'] for ann in matching_anns]
        pred_anns = coco_dt.loadAnns(dt_ann_ids)
        pred_masks = [maskUtils.decode(ann['segmentation']) for ann in pred_anns if 'segmentation' in ann]

        # Compute and save the mIoU for the current image
        mious.append(compute_miou(pred_masks, gt_masks))

    # Report mean IoU across all images
    mean_miou = np.mean(mious) if mious else 0.0  # If list is empty, return 0.0

    print(f"Mean IoU (mIoU) across all images: {mean_miou:.3f}")


def compute_iou_matrix(pred_masks, gt_masks):
    iou_matrix = np.zeros((len(pred_masks), len(gt_masks)))
    for i, pred_mask in enumerate(pred_masks):
        for j, gt_mask in enumerate(gt_masks):
            iou_matrix[i, j] = compute_iou(pred_mask, gt_mask)

    return iou_matrix


def text_similarity_bert(str1, str2):
    emb1 = get_bert_embedding(str1)
    emb2 = get_bert_embedding(str2)

    return cosine_similarity([emb1], [emb2])[0, 0]


def find_best_matches(gt_anns, gt_labels, dt_anns, dt_labels, iou_threshold, text_sim_threshold, vectorizer=None):
    best_matches = []

    # Compute pair - wise IoU
    pred_masks = [maskUtils.decode(ann['segmentation']) for ann in dt_anns]
    gt_masks = [maskUtils.decode(ann['segmentation']) for ann in gt_anns]
    ious = compute_iou_matrix(gt_masks, pred_masks)

    text_sims = np.zeros((len(gt_labels), len(dt_labels)))

    for i, gt_label in enumerate(gt_labels):
        for j, dt_label in enumerate(dt_labels):
            text_sims[i, j] = text_similarity_bert(gt_label, dt_label)

    # Find one-to-one matches satisfying both IoU and text similarity thresholds
    while ious.size > 0:
        max_iou_idx = np.unravel_index(np.argmax(ious), ious.shape)
        if ious[max_iou_idx] < iou_threshold or text_sims[max_iou_idx] < text_sim_threshold:
            break  # No admissible pair found

        best_matches.append(max_iou_idx)

        # Remove selected annotations from consideration
        ious[max_iou_idx[0], :] = 0
        ious[:, max_iou_idx[1]] = 0
        text_sims[max_iou_idx[0], :] = 0
        text_sims[:, max_iou_idx[1]] = 0

    return best_matches  # List of index pairs [(gt_idx, dt_idx), ...]


def evaluate_recall_with_mapping(coco_gt, coco_cap_gt, image_ids, pred_save_path, cap_pred_save_path, iou_threshold=0.5,
                                 text_sim_threshold=0.5):
    coco_dt = coco_gt.loadRes(pred_save_path)
    coco_cap_dt = coco_cap_gt.loadRes(cap_pred_save_path)

    true_positives = 0
    actual_positives = 0

    for image_id in tqdm(image_ids):
        try:
            # gt_ann_ids = coco_gt.getAnnIds(imgIds=image_id, iscrowd=None)
            matching_anns = [ann for ann in coco_gt.anns.values() if ann['image_id'] == image_id]
            gt_ann_ids = [ann['id'] for ann in matching_anns]
            gt_anns = coco_gt.loadAnns(gt_ann_ids)

            # dt_ann_ids = coco_dt.getAnnIds(imgIds=image_id, iscrowd=None)
            matching_anns = [ann for ann in coco_dt.anns.values() if ann['image_id'] == image_id]
            dt_ann_ids = [ann['id'] for ann in matching_anns]
            dt_anns = coco_dt.loadAnns(dt_ann_ids)

            # gt_cap_ann_ids = coco_cap_gt.getAnnIds(imgIds=image_id)
            matching_anns = [ann for ann in coco_cap_gt.anns.values() if ann['image_id'] == image_id]
            gt_cap_ann_ids = [ann['id'] for ann in matching_anns]
            gt_cap_ann = coco_cap_gt.loadAnns(gt_cap_ann_ids)[0]

            # dt_cap_ann_ids = coco_cap_dt.getAnnIds(imgIds=image_id)
            matching_anns = [ann for ann in coco_cap_dt.anns.values() if ann['image_id'] == image_id]
            dt_cap_ann_ids = [ann['id'] for ann in matching_anns]
            dt_cap_ann = coco_cap_dt.loadAnns(dt_cap_ann_ids)[0]

            gt_labels = gt_cap_ann['labels']
            dt_labels = dt_cap_ann['labels']

            actual_positives += len(gt_labels)

            # Find best matching pairs
            best_matches = find_best_matches(gt_anns, gt_labels, dt_anns, dt_labels, iou_threshold, text_sim_threshold)

            true_positives += len(best_matches)
        except Exception as e:
            print(e)

    recall = true_positives / actual_positives if actual_positives > 0 else 0

    print(f"Recall: {recall:.3f}")


def main():
    args = parse_args()

    # Set the correct split
    split = args.split
    assert split == "val" or split == "test"  # GCG Evaluation has only val and test splits
    gt_mask_path = f"{args.gt_dir_path}/{split}_gcg_coco_mask_gt.json"
    gt_cap_path = f"{args.gt_dir_path}/{split}_gcg_coco_caption_gt.json"

    print(f"Starting evalution on {split} split.")

    # Get the image names of the split
    all_images_ids = []
    with open(gt_cap_path, 'r') as f:
        contents = json.load(f)
        for image in contents['images']:
            all_images_ids.append(image['id'])

    # The directory is used to store intermediate files
    tmp_dir_path = f"tmp/{os.path.basename(args.prediction_dir_path)}_{split}"
    os.makedirs(tmp_dir_path, exist_ok=True)  # Create directory if not exists already

    # Create predictions
    pred_save_path = f"{tmp_dir_path}/mask_pred_tmp_save.json"
    cap_pred_save_path = f"{tmp_dir_path}/cap_pred_tmp_save.json"
    coco_pred_file = []
    caption_pred_dict = {}
    for image_id in all_images_ids:
        prediction_path = f"{args.prediction_dir_path}/{image_id}.json"
        with open(prediction_path, 'r') as f:
            pred = json.load(f)
            bu = pred
            key = list(pred.keys())[0]
            pred = pred[key]
            try:
                caption_pred_dict[image_id] = {'caption': pred['caption'], 'labels': pred['phrases']}
            except Exception as e:
                pred = bu
                caption_pred_dict[image_id] = {'caption': pred['caption'], 'labels': pred['phrases']}
            for rle_mask in pred['pred_masks']:
                coco_pred_file.append({"image_id": image_id, "category_id": 1, "segmentation": rle_mask, "score": 1.0})

    # Save gcg_coco_predictions
    with open(pred_save_path, 'w') as f:
        json.dump(coco_pred_file, f)

    # Prepare the CAPTION predictions in COCO format
    cap_image_ids = []
    coco_cap_pred_file = []
    for image_id, values in caption_pred_dict.items():
        cap_image_ids.append(image_id)
        coco_cap_pred_file.append({"image_id": image_id, "caption": values['caption'], "labels": values['labels']})

    # Save gcg_caption_coco_predictions
    with open(cap_pred_save_path, 'w') as f:
        json.dump(coco_cap_pred_file, f)

    # # -------------------------------#
    # 1. Evaluate AP
    # Calculate mask mAP
    # Load the ground truth and predictions in COCO format
    coco_gt = COCO(gt_mask_path)
    coco_dt = coco_gt.loadRes(pred_save_path)  # load predictions
    # Initialize COCOEval and specify the metric you want to use
    coco_eval = COCOeval(coco_gt, coco_dt, "segm")  # "segm" for segmentation
    # Evaluate on a specific category
    coco_eval.params.catIds = [1]  # your category ID
    # Evaluate
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # # -------------------------------#
    # # 2. Evaluate Caption Quality
    try:
        coco_cap_gt = COCO(gt_cap_path)
        coco_cap_result = coco_cap_gt.loadRes(cap_pred_save_path)
        # create coco_eval object by taking coco and coco_result
        coco_eval = COCOEvalCap(coco_cap_gt, coco_cap_result)
        coco_eval.params['image_id'] = coco_cap_result.getImgIds()
        coco_eval.evaluate()
        for metric, score in coco_eval.eval.items():
            print(f'{metric}: {score:.3f}')
    except:
        pass

    # # -------------------------------#
    # 3. Evaluate Mask Mean MIoU
    coco_gt = COCO(gt_mask_path)  # Load ground truth annotations
    evaluate_mask_miou(coco_gt, all_images_ids, pred_save_path)

    # # -------------------------------#
    # 4. Evaluate Recall
    evaluate_recall_with_mapping(coco_gt, coco_cap_gt, all_images_ids, pred_save_path, cap_pred_save_path,
                                 iou_threshold=0.5, text_sim_threshold=0.5)


if __name__ == "__main__":
    main()
