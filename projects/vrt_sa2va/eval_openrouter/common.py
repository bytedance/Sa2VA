"""
Shared helpers for the VER evaluation pipeline.

Used by both `eval_openrouter.py` (API-driven evaluation) and
`eval_qwen.py` (local Qwen-VL HuggingFace evaluation):

  * mask ↔ bbox helpers, IoU + Hungarian-matching scorers
  * Gemini-style box-format parsing (configurable order/normalisation)
  * SAM2 wrapper that turns a list of pred boxes into binary masks
  * per-class aggregation + LaTeX-row formatting + console summary
  * tiny .env loader and PIL→base64 utility

Anything backend-specific (prompt wording, HTTP client, HF model loader,
batched generation loop) lives in the per-backend script.
"""

from __future__ import annotations

import base64
import os
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment


# ────────────────────────── env / .env loader ────────────────────────────

def load_env_from_file(env_path: str) -> None:
    """Lightweight .env loader (avoids pulling python-dotenv as a hard dep)."""
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val


# ─────────────────────── PIL ↔ base64 (HTTP payload) ─────────────────────

def encode_pil_to_base64(image: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img = image.convert("RGB") if image.mode != "RGB" else image
    img.save(buf, format=fmt, quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─────────────────────── mask ↔ bbox helpers ─────────────────────────────

def mask_to_bbox(mask: np.ndarray) -> List[int]:
    """Binary HxW mask → tight [xmin, ymin, xmax, ymax] (0-box for empty mask)."""
    if mask.sum() == 0:
        return [0, 0, 0, 0]
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return [int(xmin), int(ymin), int(xmax), int(ymax)]


def bbox_iou(a: List[int], b: List[int]) -> float:
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def calculate_miou_bbox(pred_boxes: List[List[int]], gt_boxes: List[List[int]]) -> List[float]:
    """Hungarian-match pred boxes to GT boxes; per-GT IoU (0 if unmatched)."""
    if not gt_boxes:
        return []
    if not pred_boxes:
        return [0.0] * len(gt_boxes)
    iou = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, pb in enumerate(pred_boxes):
        for j, gb in enumerate(gt_boxes):
            iou[i, j] = bbox_iou(list(pb), list(gb))
    row_ind, col_ind = linear_sum_assignment(1 - iou)
    out = np.zeros(len(gt_boxes), dtype=np.float32)
    out[col_ind] = iou[row_ind, col_ind]
    return out.tolist()


def calculate_miou_mask(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]) -> List[float]:
    """Hungarian-match predicted masks to GT masks; per-GT IoU (0 if unmatched)."""
    if not gt_masks:
        return []
    if not pred_masks:
        return [0.0] * len(gt_masks)

    pm = np.stack([m.astype(bool) for m in pred_masks])
    gm = np.stack([m.astype(bool) for m in gt_masks])
    inter = np.logical_and(pm[:, None], gm[None, :]).sum(axis=(2, 3)).astype(np.float32)
    union = np.logical_or(pm[:, None], gm[None, :]).sum(axis=(2, 3)).astype(np.float32)
    iou = np.zeros_like(inter)
    np.divide(inter, union, out=iou, where=union > 0)

    row_ind, col_ind = linear_sum_assignment(1 - iou)
    out = np.zeros(len(gt_masks), dtype=np.float32)
    out[col_ind] = iou[row_ind, col_ind]
    return out.tolist()


def encode_mask_rle(mask: np.ndarray) -> Dict:
    """Binary HxW mask → COCO RLE (counts as utf-8 string for JSON)."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def decode_mask_rle(rle: Dict) -> np.ndarray:
    return mask_utils.decode(rle).astype(bool)


# ─────────────────── prediction parsing (configurable) ───────────────────

# Supported box conventions:
#   yxyx_norm1000 — [ymin, xmin, ymax, xmax] normalised 0-1000 (Gemini default)
#   xyxy_norm1000 — [x1, y1, x2, y2]         normalised 0-1000 (Qwen instructed)
#   xyxy_pixel    — [x1, y1, x2, y2]         raw pixel        (Qwen native fallback)
BOX_FORMATS = ("yxyx_norm1000", "xyxy_norm1000", "xyxy_pixel")

_BBOX_PATTERN = re.compile(r"\[([\d\s,.\-]+)\]")


def auto_box_format_for_model(model_id: str) -> str:
    """Pick a sensible default convention based on the model family."""
    m = model_id.lower()
    if m.startswith("google/") or "gemini" in m:
        return "yxyx_norm1000"
    if m.startswith("qwen/") or "qwen" in m:
        return "xyxy_norm1000"
    return "yxyx_norm1000"


def extract_bboxes_from_text(text: str, width: int, height: int,
                             box_format: str = "yxyx_norm1000") -> List[List[int]]:
    """
    Parse every `[a, b, c, d]` group in `text` according to `box_format`,
    returning pixel `[xmin, ymin, xmax, ymax]` boxes.

    Robustness shortcuts:
      * 0-1 normalisation auto-rescaled to 0-1000 (some models do this).
      * For *_norm1000 formats: if values clearly exceed 1000, fall back to
        pixel interpretation (Qwen sometimes ignores normalise instructions).
      * Degenerate / empty boxes dropped.
    """
    if box_format not in BOX_FORMATS:
        raise ValueError(f"Unknown box_format {box_format!r}; expected one of {BOX_FORMATS}")

    bboxes: List[List[int]] = []
    for match in _BBOX_PATTERN.findall(text):
        try:
            coords = [float(c.strip()) for c in match.split(",")]
        except ValueError:
            continue
        if len(coords) != 4:
            continue
        a, b, c, d = coords
        max_val = max(a, b, c, d)

        if max_val <= 1.5 and box_format != "xyxy_pixel":
            a, b, c, d = (v * 1000 for v in (a, b, c, d))
            max_val = max(a, b, c, d)

        if box_format == "yxyx_norm1000":
            ymin_n, xmin_n, ymax_n, xmax_n = a, b, c, d
            xmin = int(round(xmin_n / 1000 * width))
            ymin = int(round(ymin_n / 1000 * height))
            xmax = int(round(xmax_n / 1000 * width))
            ymax = int(round(ymax_n / 1000 * height))
        elif box_format == "xyxy_norm1000":
            if max_val > 1000.0 and (a < width * 1.1 and c < width * 1.1
                                     and b < height * 1.1 and d < height * 1.1):
                xmin, ymin, xmax, ymax = int(a), int(b), int(c), int(d)
            else:
                x1_n, y1_n, x2_n, y2_n = a, b, c, d
                xmin = int(round(x1_n / 1000 * width))
                ymin = int(round(y1_n / 1000 * height))
                xmax = int(round(x2_n / 1000 * width))
                ymax = int(round(y2_n / 1000 * height))
        else:  # xyxy_pixel
            xmin, ymin, xmax, ymax = int(a), int(b), int(c), int(d)

        xmin = max(0, min(xmin, width - 1))
        ymin = max(0, min(ymin, height - 1))
        xmax = max(0, min(xmax, width))
        ymax = max(0, min(ymax, height))
        if xmax <= xmin or ymax <= ymin:
            continue
        bboxes.append([xmin, ymin, xmax, ymax])
    return bboxes


def separate_bboxes_by_section(text: str, width: int, height: int,
                               box_format: str = "yxyx_norm1000"
                               ) -> Tuple[List[List[int]], List[List[int]]]:
    """Split predicted boxes into reasoning (`<think>`) vs answer (`<answer>`) buckets."""
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if answer_match:
        reasoning_text = text[: answer_match.start()]
        answer_text = answer_match.group(1)
    elif "<think>" in text and "</think>" in text:
        reasoning_text = text.split("</think>")[0]
        answer_text = text.split("</think>")[-1]
    elif "```json" in text:
        reasoning_text = text.split("```json")[0]
        answer_text = text.split("```json", 1)[-1]
    else:
        reasoning_text, answer_text = text, text

    r_boxes = extract_bboxes_from_text(reasoning_text, width, height, box_format)
    a_boxes = extract_bboxes_from_text(answer_text, width, height, box_format)
    r_boxes = [list(b) for b in {tuple(b) for b in r_boxes}]
    a_boxes = [list(b) for b in {tuple(b) for b in a_boxes}]
    return r_boxes, a_boxes


# ────────────────────────── per-sample record ────────────────────────────

def build_record(sample: Dict, pred_text: str, box_format: str) -> Dict:
    """
    Build a Phase-1 record from a packed VRT sample + raw model response.

    The record carries:
      * sample_info        — key, question, class_ids, image dims
      * ground_truth       — GT obj ids, tight GT bboxes, GT masks (RLE)
      * prediction         — raw text, parsed pred bboxes (mask RLEs filled
                             in later by `score_with_sam2`)
      * evaluation_bbox    — bbox-vs-bbox mIoU (always available after parse)
      * evaluation         — mask-vs-mask mIoU (filled by `score_with_sam2`)
    """
    image: Image.Image = sample["image"]
    objects_info: Dict[int, Dict] = sample.get("objects_info", {})
    gt_r_ids: List[int] = list(sample.get("human_labeled_r_objs", []))
    gt_a_ids: List[int] = list(sample.get("human_labeled_a_objs", []))

    gt_r_masks_np = [objects_info[i]["mask"] for i in gt_r_ids if i in objects_info]
    gt_a_masks_np = [objects_info[i]["mask"] for i in gt_a_ids if i in objects_info]
    gt_r_boxes = [mask_to_bbox(m) for m in gt_r_masks_np]
    gt_a_boxes = [mask_to_bbox(m) for m in gt_a_masks_np]
    gt_r_masks_rle = [encode_mask_rle(m) for m in gt_r_masks_np]
    gt_a_masks_rle = [encode_mask_rle(m) for m in gt_a_masks_np]

    pred_r_boxes, pred_a_boxes = separate_bboxes_by_section(
        pred_text, image.width, image.height, box_format)
    r_box_ious = calculate_miou_bbox(pred_r_boxes, gt_r_boxes)
    a_box_ious = calculate_miou_bbox(pred_a_boxes, gt_a_boxes)

    return {
        "sample_info": {
            "key": sample["key"],
            "question": sample["question"],
            "class_ids": list(sample.get("class_ids", [])),
            "image_size": [image.width, image.height],
        },
        "ground_truth": {
            "reasoning_obj_ids": gt_r_ids,
            "answer_obj_ids": gt_a_ids,
            "reasoning_bboxes": gt_r_boxes,
            "answer_bboxes": gt_a_boxes,
            "reasoning_masks_rle": gt_r_masks_rle,
            "answer_masks_rle": gt_a_masks_rle,
        },
        "prediction": {
            "text": pred_text,
            "reasoning_bboxes": pred_r_boxes,
            "answer_bboxes": pred_a_boxes,
            "reasoning_masks_rle": [],
            "answer_masks_rle": [],
        },
        "evaluation_bbox": {
            "reasoning_ious": r_box_ious,
            "answer_ious": a_box_ious,
            "reasoning_miou": float(np.mean(r_box_ious)) if r_box_ious else 0.0,
            "answer_miou": float(np.mean(a_box_ious)) if a_box_ious else 0.0,
        },
        "evaluation": {
            "reasoning_ious": [],
            "answer_ious": [],
            "reasoning_miou": 0.0,
            "answer_miou": 0.0,
        },
    }


# ────────────────────────────── SAM2 ─────────────────────────────────────

class SAM2BoxToMask:
    """Wraps SAM2ImagePredictor — set_image once per sample, predict once per box.
    Returns one binary HxW mask per input box."""

    def __init__(self, checkpoint: str, config: str, device: Optional[str] = None):
        import torch
        from third_parts.sam2.build_sam import build_sam2
        from third_parts.sam2.sam2_image_predictor import SAM2ImagePredictor

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[sam2] loading {checkpoint} on {device}")
        sam2_model = build_sam2(config, checkpoint, device=torch.device(device))
        self.predictor = SAM2ImagePredictor(sam2_model)

        if device.startswith("cuda"):
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        print("[sam2] ready")

    def boxes_to_masks(self, image_np: np.ndarray, boxes: List[List[int]]) -> List[np.ndarray]:
        if not boxes:
            return []
        self.predictor.set_image(image_np)
        out: List[np.ndarray] = []
        for box in boxes:
            masks, _scores, _ = self.predictor.predict(
                point_coords=None, point_labels=None,
                box=np.asarray(box, dtype=np.float32),
                multimask_output=False,
            )
            out.append(np.asarray(masks[0]).astype(bool))
        return out


def score_with_sam2(record: Dict, dataset, sam2: SAM2BoxToMask) -> Dict:
    """Phase-2 in-place augmentation: SAM2 the predicted boxes, score
    against GT masks. Modifies and returns `record`."""
    sample = dataset.get_sample(record["sample_info"]["key"])
    if sample is None:
        return record

    image_np = np.array(sample["image"].convert("RGB"))
    pred_r = sam2.boxes_to_masks(image_np, record["prediction"]["reasoning_bboxes"])
    pred_a = sam2.boxes_to_masks(image_np, record["prediction"]["answer_bboxes"])
    record["prediction"]["reasoning_masks_rle"] = [encode_mask_rle(m) for m in pred_r]
    record["prediction"]["answer_masks_rle"] = [encode_mask_rle(m) for m in pred_a]

    gt_r = [decode_mask_rle(r) for r in record["ground_truth"]["reasoning_masks_rle"]]
    gt_a = [decode_mask_rle(r) for r in record["ground_truth"]["answer_masks_rle"]]
    r_ious = calculate_miou_mask(pred_r, gt_r)
    a_ious = calculate_miou_mask(pred_a, gt_a)
    record["evaluation"]["reasoning_ious"] = r_ious
    record["evaluation"]["answer_ious"] = a_ious
    record["evaluation"]["reasoning_miou"] = float(np.mean(r_ious)) if r_ious else 0.0
    record["evaluation"]["answer_miou"] = float(np.mean(a_ious)) if a_ious else 0.0
    return record


# ───────────── per-class aggregation (table-ready output) ───────────────

# Matches the LaTeX table column order in the paper.
TABLE_CLASSES: List[str] = ["#comp", "#func", "#loc", "#visf"]


def _lq_sq(ious: List[float]) -> Tuple[float, float, float]:
    """(mIoU, LQ, SQ): mIoU = mean; LQ = hit-rate at IoU>0.5; SQ = mean among hits."""
    if not ious:
        return 0.0, 0.0, 0.0
    arr = np.asarray(ious, dtype=np.float32)
    miou = float(arr.mean())
    hits = arr > 0.5
    lq = float(hits.sum() / len(arr))
    sq = float(arr[hits].mean()) if hits.any() else 0.0
    return miou, lq, sq


def _aggregate_one_metric(results: List[Dict], eval_key: str) -> Dict:
    """Per-class + overall mIoU/LQ/SQ for a single eval namespace
    ('evaluation' for masks, 'evaluation_bbox' for boxes)."""
    buckets: Dict[str, Dict[str, List[float]]] = {
        c: {"r": [], "a": []} for c in TABLE_CLASSES
    }
    buckets["overall"] = {"r": [], "a": []}

    for r in results:
        ev = r.get(eval_key, {})
        r_ious = ev.get("reasoning_ious", [])
        a_ious = ev.get("answer_ious", [])
        cls_ids = r["sample_info"].get("class_ids", [])
        for c in list(cls_ids) + ["overall"]:
            if c not in buckets:
                continue
            buckets[c]["r"].extend(r_ious)
            buckets[c]["a"].extend(a_ious)

    out: Dict[str, Dict] = {}
    for c, b in buckets.items():
        r_miou, r_lq, r_sq = _lq_sq(b["r"])
        a_miou, a_lq, a_sq = _lq_sq(b["a"])
        out[c] = {
            "reasoning_miou": r_miou, "reasoning_lq": r_lq, "reasoning_sq": r_sq,
            "answer_miou": a_miou, "answer_lq": a_lq, "answer_sq": a_sq,
            "n_reasoning": len(b["r"]), "n_answer": len(b["a"]),
        }
    return out


def aggregate_metrics(results: List[Dict], model_label: str = "model") -> Dict:
    """
    Returns a summary dict with both the flat overall-mask `metrics` field
    (kept for backward compatibility with earlier consumers) and per-class
    breakdowns for both box and mask versions, plus LaTeX-ready row strings
    labelled with `model_label`.
    """
    mask_breakdown = _aggregate_one_metric(results, "evaluation")
    bbox_breakdown = _aggregate_one_metric(results, "evaluation_bbox")
    overall_mask = mask_breakdown["overall"]

    return {
        "metrics": {
            "reasoning_miou": overall_mask["reasoning_miou"],
            "reasoning_lq": overall_mask["reasoning_lq"],
            "reasoning_sq": overall_mask["reasoning_sq"],
            "answer_miou": overall_mask["answer_miou"],
            "answer_lq": overall_mask["answer_lq"],
            "answer_sq": overall_mask["answer_sq"],
        },
        "total_reasoning_masks_evaluated": overall_mask["n_reasoning"],
        "total_answer_masks_evaluated": overall_mask["n_answer"],
        "per_class_breakdown": {
            "bbox": bbox_breakdown,
            "mask": mask_breakdown,
        },
        "latex_rows": _build_latex_rows(model_label, bbox_breakdown, mask_breakdown),
    }


def _build_latex_rows(model_label: str, bbox_b: Dict, mask_b: Dict) -> Dict[str, str]:
    """Format the four #cls + Overall triplet (VRQ-C, VRQ-G, mIoU) into a
    LaTeX-ready row for each metric kind. VRQ-C ≡ reasoning_lq,
    VRQ-G ≡ reasoning_sq, mIoU ≡ answer_miou — matches the paper table."""
    def _row(label: str, br: Dict) -> str:
        cells = [label]
        for c in TABLE_CLASSES + ["overall"]:
            m = br.get(c, {})
            cells.append(
                f"{m.get('reasoning_lq', 0.0) * 100:.1f} & "
                f"{m.get('reasoning_sq', 0.0) * 100:.1f} & "
                f"{m.get('answer_miou', 0.0) * 100:.1f}"
            )
        return " & ".join(cells) + r" \\"
    return {
        "bbox": _row(f"{model_label} (box)", bbox_b),
        "mask": _row(f"{model_label} + SAM2", mask_b),
    }


def _print_breakdown(title: str, breakdown: Dict) -> None:
    print(f"\n{title}")
    print(f"  {'class':>8s}  {'VRQ-C':>6s}  {'VRQ-G':>6s}  {'mIoU':>6s}  (n_R / n_A)")
    for c in TABLE_CLASSES + ["overall"]:
        m = breakdown.get(c, {})
        print(f"  {c:>8s}  {m.get('reasoning_lq', 0)*100:>6.1f}  "
              f"{m.get('reasoning_sq', 0)*100:>6.1f}  "
              f"{m.get('answer_miou', 0)*100:>6.1f}  "
              f"({m.get('n_reasoning', 0)} / {m.get('n_answer', 0)})")


def print_summary(summary: Dict) -> None:
    print("\n=== VER Eval Summary ===")
    print(f"Model: {summary.get('model')}")
    print(f"Total samples: {summary['total_samples']}")
    print(f"R masks: {summary['total_reasoning_masks_evaluated']}, "
          f"A masks: {summary['total_answer_masks_evaluated']}")

    pcb = summary.get("per_class_breakdown", {})
    if "bbox" in pcb:
        _print_breakdown("Box-vs-box (no SAM2):", pcb["bbox"])
    if "mask" in pcb:
        _print_breakdown("Mask-vs-mask (SAM2 on preds, human masks on GT):", pcb["mask"])

    rows = summary.get("latex_rows", {})
    if rows:
        print("\nLaTeX rows (VRQ-C & VRQ-G & mIoU per class, then overall):")
        for v in rows.values():
            print(f"  {v}")
