"""
Visualise OpenRouter / Gemini + SAM2 VER eval results.

Reads `<output_dir>/ver_results.json` (or the older `ver_bbox_results.json`)
produced by `eval_openrouter.py` and the packed VRT eval tfrecord
(for the original image + GT masks), then emits one PNG per sample into
`<output_dir>/visualizations/`.

Layout per sample (matplotlib):

    ┌───────────────────────────┬──────────────────────────────────┐
    │ Prediction (boxes only)   │ Ground Truth (masks only)        │
    │  image with pred boxes    │  image with GT mask overlays     │
    │  coloured green ≥0.5 IoU  │  (no boxes, no SAM2 — these are  │
    │  red < 0.5 IoU            │  the human-labelled masks)       │
    ├───────────────────────────┴──────────────────────────────────┤
    │ <think> text (cleaned, with bbox literals highlighted)       │
    │ <answer> text                                                │
    └──────────────────────────────────────────────────────────────┘

Usage
-----
    PYTHONPATH=. uv run --extra latest python \\
        projects/vrt_sa2va/eval_openrouter/visualize_results.py \\
        workspace/eval_results/openrouter_gemini_2.5_pro_smoke10

Filtering
---------
By default visualises every sample. Pass `--low-iou-only` to skip samples
whose mean IoU exceeds 0.5 (useful for cherry-picking failures).
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from pycocotools import mask as mask_utils

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from projects.vrt_sa2va.evaluation.packed_vrt_eval_dataloader import PackedVRTEvalDataset


# ─────────────────────────── colour helpers ──────────────────────────────

# Distinct hues for indexing reasoning vs answer boxes
PALETTE = [
    (0.90, 0.30, 0.30),   # red
    (0.30, 0.55, 0.85),   # blue
    (0.95, 0.70, 0.20),   # amber
    (0.40, 0.75, 0.40),   # green
    (0.70, 0.40, 0.85),   # purple
    (0.25, 0.75, 0.75),   # teal
    (0.95, 0.55, 0.30),   # orange
    (0.55, 0.55, 0.55),   # gray
]


def iou_colour(iou: float) -> Tuple[float, float, float]:
    """Green (good) → red (bad) interpolation around the 0.5 hit threshold."""
    if iou >= 0.5:
        return (0.20, 0.75, 0.20)
    if iou >= 0.25:
        return (0.95, 0.65, 0.20)
    return (0.85, 0.20, 0.20)


# ───────────────── prediction text cleaning + rendering ──────────────────

_TAG_PATTERN = re.compile(r"<\|im_end\|>|<obj\d+>\s*")
_BBOX_INLINE_PATTERN = re.compile(r"\[\s*\d[\d\s,.\-]*\d\s*\]")


def clean_text(text: str) -> str:
    text = _TAG_PATTERN.sub("", text)
    return text.strip()


def render_text_panel(ax, raw_text: str, title: str, fontsize: int = 9):
    """
    Draw `raw_text` into the given matplotlib axes, highlighting any inline
    `[a, b, c, d]` bbox literals so the reader can spot them at a glance.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title(title, fontweight="bold", fontsize=fontsize + 2, loc="left", pad=4)
    text = clean_text(raw_text) or "(empty)"

    # word-wrap by character count (matplotlib doesn't word-wrap automatically inside transAxes)
    MAX_CHARS = 110
    parts = _BBOX_INLINE_PATTERN.split(text)
    bbox_tokens = _BBOX_INLINE_PATTERN.findall(text)

    # Re-stitch into (string, is_bbox) runs
    runs: List[Tuple[str, bool]] = []
    for i, p in enumerate(parts):
        if p:
            runs.append((p, False))
        if i < len(bbox_tokens):
            runs.append((bbox_tokens[i], True))

    # Tokenise into words / spaces / newlines
    tokens: List[Tuple[str, bool]] = []  # (text, is_bbox)
    for run_str, is_bbox in runs:
        if is_bbox:
            tokens.append((run_str, True))
        else:
            for line_idx, line in enumerate(run_str.split("\n")):
                if line_idx > 0:
                    tokens.append(("\n", False))
                for word_idx, word in enumerate(line.split(" ")):
                    if word_idx > 0:
                        tokens.append((" ", False))
                    if word:
                        tokens.append((word, False))

    # Pack into display lines
    lines: List[List[Tuple[str, bool]]] = []
    cur: List[Tuple[str, bool]] = []
    cur_len = 0
    for tok, is_bbox in tokens:
        if tok == "\n":
            lines.append(cur)
            cur, cur_len = [], 0
            continue
        if cur_len + len(tok) > MAX_CHARS and cur_len > 0 and not is_bbox:
            lines.append(cur)
            cur, cur_len = [], 0
            if tok == " ":
                continue
        cur.append((tok, is_bbox))
        cur_len += len(tok)
    if cur:
        lines.append(cur)

    # Render
    fig = ax.get_figure()
    ax_pos = ax.get_position()
    ax_w_in = ax_pos.width * fig.get_figwidth()
    char_w_in = fontsize * 0.6 / 72.0
    char_frac = (char_w_in / ax_w_in) if ax_w_in > 0 else 0.008

    n = max(len(lines), 1)
    line_h = min(0.07, 0.96 / n)
    x0, y0 = 0.01, 0.96
    for li, line_tokens in enumerate(lines):
        y = y0 - li * line_h
        if y < 0.02:
            ax.text(x0, 0.01, "…", transform=ax.transAxes,
                    fontsize=fontsize, color="gray", va="bottom")
            break
        x = x0
        for tok, is_bbox in line_tokens:
            if not tok:
                continue
            w = len(tok) * char_frac
            if is_bbox:
                ax.text(x, y, tok, transform=ax.transAxes, fontsize=fontsize,
                        va="top", ha="left", fontfamily="monospace",
                        fontweight="bold", color=(0.10, 0.30, 0.55),
                        bbox=dict(facecolor=(0.85, 0.92, 1.0),
                                  edgecolor=(0.30, 0.55, 0.85),
                                  boxstyle="round,pad=0.15", linewidth=1.0))
            else:
                ax.text(x, y, tok, transform=ax.transAxes, fontsize=fontsize,
                        va="top", ha="left", fontfamily="monospace", color="black")
            x += w


# ───────────────────── bbox + mask drawing helpers ───────────────────────

def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: Tuple[float, float, float],
                 alpha: float = 0.45) -> np.ndarray:
    """Tint mask region of `rgb` (uint8 HxWx3) with `color` (0-1)."""
    if mask.sum() == 0:
        return rgb
    out = rgb.astype(np.float32)
    cm = np.array(color, dtype=np.float32) * 255.0
    m = mask.astype(bool)
    out[m] = out[m] * (1 - alpha) + cm * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_boxes(ax, boxes: List[List[int]], colors: List[Tuple[float, float, float]],
               labels: List[str], linewidth: float = 2.5):
    for box, color, label in zip(boxes, colors, labels):
        x1, y1, x2, y2 = box
        rect = Rectangle((x1, y1), max(1, x2 - x1), max(1, y2 - y1),
                         linewidth=linewidth, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        if label:
            ax.text(x1 + 2, max(2, y1 - 4), label, fontsize=8, color="white",
                    bbox=dict(facecolor=color, edgecolor="none",
                              boxstyle="round,pad=0.15"))


# ─────────────────────────── per-sample render ───────────────────────────

def _gt_masks_for_record(rec: Dict, dataset_sample: Dict
                         ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract GT reasoning + answer masks from either the record (RLE)
    or fall back to the dataset sample's objects_info."""
    gt = rec["ground_truth"]
    if gt.get("reasoning_masks_rle") or gt.get("answer_masks_rle"):
        r = [mask_utils.decode(rle).astype(bool) for rle in gt.get("reasoning_masks_rle", [])]
        a = [mask_utils.decode(rle).astype(bool) for rle in gt.get("answer_masks_rle", [])]
        return r, a
    objects_info = dataset_sample.get("objects_info", {})
    r = [objects_info[i]["mask"].astype(bool)
         for i in gt.get("reasoning_obj_ids", []) if i in objects_info]
    a = [objects_info[i]["mask"].astype(bool)
         for i in gt.get("answer_obj_ids", []) if i in objects_info]
    return r, a


def render_sample(sample_record: Dict, dataset_sample: Dict, save_path: str):
    image: Image.Image = dataset_sample["image"]

    pred_r = sample_record["prediction"]["reasoning_bboxes"]
    pred_a = sample_record["prediction"]["answer_bboxes"]
    gt_r_ids = sample_record["ground_truth"]["reasoning_obj_ids"]
    gt_a_ids = sample_record["ground_truth"]["answer_obj_ids"]
    # Prefer mask IoU; fall back to box IoU when SAM2 phase was skipped.
    ev = sample_record.get("evaluation", {}) or {}
    r_ious = ev.get("reasoning_ious") or []
    a_ious = ev.get("answer_ious") or []
    r_miou = ev.get("reasoning_miou", 0.0)
    a_miou = ev.get("answer_miou", 0.0)
    iou_kind = "mask"
    if not r_ious and not a_ious:
        ev_b = sample_record.get("evaluation_bbox", {}) or {}
        r_ious = ev_b.get("reasoning_ious") or []
        a_ious = ev_b.get("answer_ious") or []
        r_miou = ev_b.get("reasoning_miou", 0.0)
        a_miou = ev_b.get("answer_miou", 0.0)
        iou_kind = "box"
    gt_r_masks, gt_a_masks = _gt_masks_for_record(sample_record, dataset_sample)

    img_np = np.array(image.convert("RGB"))

    # Right-side composite: original image with GT masks tinted on top.
    # Reasoning masks each get a distinct palette colour; answer mask is blue.
    gt_overlay = img_np.copy()
    for i, m in enumerate(gt_r_masks):
        gt_overlay = overlay_mask(gt_overlay, m, PALETTE[i % len(PALETTE)], alpha=0.45)
    for m in gt_a_masks:
        gt_overlay = overlay_mask(gt_overlay, m, (0.10, 0.55, 0.95), alpha=0.55)

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"{sample_record['sample_info']['key']}   "
        f"R-mIoU={r_miou:.3f}   A-mIoU={a_miou:.3f}   ({iou_kind})",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(0.5, 0.965,
             f"Q: {sample_record['sample_info']['question']}",
             ha="center", va="top", fontsize=10, wrap=True,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.7))

    gs = fig.add_gridspec(2, 2, height_ratios=[5, 3],
                          top=0.94, bottom=0.01, left=0.02, right=0.99,
                          hspace=0.10, wspace=0.05)

    # ── Top-left: Prediction (boxes only) ──────────────────────────────
    ax_p = fig.add_subplot(gs[0, 0])
    ax_p.imshow(img_np)
    ax_p.set_title("Prediction (boxes)", fontweight="bold", fontsize=12)
    ax_p.axis("off")
    # Per-pred IoU is *best vs any GT mask* (display only — actual scoring
    # uses Hungarian matching of SAM2-derived pred masks vs GT masks).
    def _box_mask_iou(box, m):
        x1, y1, x2, y2 = box
        h, w = m.shape
        x1 = max(0, min(int(x1), w - 1)); x2 = max(0, min(int(x2), w))
        y1 = max(0, min(int(y1), h - 1)); y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return 0.0
        inter = m[y1:y2, x1:x2].sum()
        union = m.sum() + (x2 - x1) * (y2 - y1) - inter
        return float(inter) / float(union) if union > 0 else 0.0

    def _best(boxes, masks):
        return [max((_box_mask_iou(b, m) for m in masks), default=0.0) for b in boxes]

    p_r_best = _best(pred_r, gt_r_masks)
    p_a_best = _best(pred_a, gt_a_masks)
    draw_boxes(ax_p, pred_r,
               [iou_colour(v) for v in p_r_best],
               [f"R{i+1} IoU={v:.2f}" for i, v in enumerate(p_r_best)])
    draw_boxes(ax_p, pred_a,
               [iou_colour(v) for v in p_a_best],
               [f"A{i+1} IoU={v:.2f}" for i, v in enumerate(p_a_best)],
               linewidth=3.5)

    # ── Top-right: Ground Truth (masks only) ───────────────────────────
    ax_g = fig.add_subplot(gs[0, 1])
    ax_g.imshow(gt_overlay)
    ax_g.set_title("Ground Truth (masks)", fontweight="bold", fontsize=12)
    ax_g.axis("off")
    # Tiny per-mask centroid label so reader can match GT mask → IoU score.
    for i, m in enumerate(gt_r_masks):
        if m.any():
            ys, xs = np.nonzero(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            iou = r_ious[i] if i < len(r_ious) else 0.0
            ax_g.text(cx, cy, f"R{i+1} obj{gt_r_ids[i]}\nIoU={iou:.2f}",
                       fontsize=8, color="white", ha="center", va="center",
                       bbox=dict(facecolor=PALETTE[i % len(PALETTE)],
                                 edgecolor="none", boxstyle="round,pad=0.2", alpha=0.85))
    for i, m in enumerate(gt_a_masks):
        if m.any():
            ys, xs = np.nonzero(m)
            cx, cy = int(xs.mean()), int(ys.mean())
            iou = a_ious[i] if i < len(a_ious) else 0.0
            ax_g.text(cx, cy, f"A obj{gt_a_ids[i]}\nIoU={iou:.2f}",
                       fontsize=9, color="white", ha="center", va="center",
                       bbox=dict(facecolor=(0.10, 0.55, 0.95),
                                 edgecolor="none", boxstyle="round,pad=0.2", alpha=0.9))

    # ── Bottom: text panels ─────────────────────────────────────────────
    ax_think = fig.add_subplot(gs[1, 0])
    ax_think.set_facecolor("#f9f9f9")
    text = sample_record["prediction"]["text"] or ""
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    think_str = think_match.group(1) if think_match else (
        text.split("<answer>")[0] if "<answer>" in text else text)
    answer_str = answer_match.group(1) if answer_match else (
        text.split("</think>")[-1] if "</think>" in text else "")
    render_text_panel(ax_think, think_str, "<think>", fontsize=9)

    ax_ans = fig.add_subplot(gs[1, 1])
    ax_ans.set_facecolor("#f0f8ff")
    render_text_panel(ax_ans, answer_str, "<answer>", fontsize=9)

    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────── driver ──────────────────────────────────

def visualize(results_dir: str, packed_tfrecord: str,
              low_iou_only: bool = False,
              max_samples: Optional[int] = None) -> None:
    # Try the mask-eval filename, then the older bbox one, then the
    # summary file's `detailed_results` key.
    candidates = [
        ("ver_results.json", None),
        ("ver_bbox_results.json", None),
        ("ver_summary.json", "detailed_results"),
        ("ver_bbox_summary.json", "detailed_results"),
    ]
    results = None
    for fname, sub in candidates:
        path = os.path.join(results_dir, fname)
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            results = data if sub is None else data.get(sub, [])
            print(f"[viz] loaded {len(results)} records from {path}")
            break
    if results is None:
        raise FileNotFoundError(
            f"No ver_results.json / ver_bbox_results.json found under {results_dir}")

    print(f"[viz] {len(results)} sample records loaded from {results_dir}")

    out_dir = os.path.join(results_dir, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[viz] loading packed dataset: {packed_tfrecord}")
    dataset = PackedVRTEvalDataset(packed_tfrecord)

    # Sort for stable output ordering
    results = sorted(results, key=lambda r: r["sample_info"]["key"])

    rendered = 0
    for rec in results:
        if max_samples is not None and rendered >= max_samples:
            break
        if low_iou_only:
            mean_iou = (rec["evaluation"]["reasoning_miou"]
                        + rec["evaluation"]["answer_miou"]) / 2.0
            if mean_iou >= 0.5:
                continue

        key = rec["sample_info"]["key"]
        ds_sample = dataset.get_sample(key)
        if ds_sample is None:
            print(f"[viz][warn] dataset miss for {key}, skipping")
            continue

        out_path = os.path.join(out_dir, f"{key}.png")
        try:
            render_sample(rec, ds_sample, out_path)
            rendered += 1
        except Exception as e:  # noqa: BLE001
            print(f"[viz][warn] {key} render failed: {e!r}")

    print(f"[viz] {rendered} visualisations written to {out_dir}")


def parse_args():
    p = argparse.ArgumentParser(description="Visualise OpenRouter VER bbox eval results.")
    p.add_argument("results_dir",
                   help="Eval output dir (containing ver_bbox_results.json).")
    p.add_argument("--packed_tfrecord",
                   default=os.path.join(PROJECT_ROOT, "data/VRT-Eval/vrt_eval.tfrecord"),
                   help="Path to packed VRT eval TFRecord.")
    p.add_argument("--low-iou-only", action="store_true",
                   help="Only render samples whose mean IoU < 0.5.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap number of samples rendered (debug).")
    return p.parse_args()


def main():
    args = parse_args()
    visualize(args.results_dir, args.packed_tfrecord,
              low_iou_only=args.low_iou_only,
              max_samples=args.max_samples)


if __name__ == "__main__":
    main()
