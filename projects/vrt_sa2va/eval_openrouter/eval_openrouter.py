"""
VER mask evaluation via OpenRouter (Gemini, GPT, OpenRouter-hosted Qwen, etc.).

Pipeline
--------
1. Inference phase (parallel HTTP via OpenRouter) — for each sample, prompt the
   model with the strict VER prompt and parse out bounding boxes.
2. SAM2 phase (sequential, GPU) — feed each predicted box into SAM2 to obtain
   a binary mask.
3. Scoring — Hungarian-match predicted *masks* against the human-labelled GT
   masks shipped with the packed dataset (SAM2 is NOT involved on the GT side)
   and compute mIoU / LQ / SQ for both reasoning and answer sections.

Usage
-----
    # 1. Put OPENROUTER_API_KEY=sk-or-... in <project_root>/.env
    # 2. Run:
    PYTHONPATH=. uv run --extra latest python \\
        projects/vrt_sa2va/eval_openrouter/eval_openrouter.py \\
        --output_dir workspace/eval_results/openrouter_gemini_3.1_pro_full \\
        --model google/gemini-3.1-pro-preview --workers 8 --visualize

Concurrency
-----------
`--workers N` issues N requests in parallel (OpenRouter rate-limits per model;
4-8 is usually safe). The SAM2 phase runs sequentially on a single GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import tqdm
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from projects.vrt_sa2va.evaluation.packed_vrt_eval_dataloader import PackedVRTEvalDataset
from projects.vrt_sa2va.eval_openrouter.common import (
    BOX_FORMATS,
    SAM2BoxToMask,
    aggregate_metrics,
    auto_box_format_for_model,
    build_record,
    encode_pil_to_base64,
    load_env_from_file,
    print_summary,
    score_with_sam2,
)


# ───────────────────────── strict VER prompt ─────────────────────────────

# Prompt design notes:
#
# The VER benchmark labels two distinct object sets per sample:
#   * reasoning evidence — every object referenced while reasoning to the
#     answer (typically 2–4 per sample; scored by `reasoning_miou`)
#   * answer objects — only the object(s) that directly answer the question
#     (typically 1; scored by `answer_miou`)
#
# We map the training tag convention (<ver>...</ver> for evidence,
# <vea>...</vea> for the answer) to box output: box after every evidence
# reference inside <think>, box only after the answer object(s) inside
# <answer>. The instructions below make this split explicit, since otherwise
# vision LLMs tend to (a) only emit the answer's box and forget evidence,
# (b) duplicate every <think> box into <answer>, or (c) wrap predictions in
# an extra ```json``` block.

_BOX_FORMAT_DESC = {
    "yxyx_norm1000": (
        "[ymin, xmin, ymax, xmax]",
        "with each coordinate normalised to the range 0–1000 (integers)",
        "[120, 45, 600, 400]",
    ),
    "xyxy_norm1000": (
        "[x1, y1, x2, y2]",
        "with each coordinate normalised to the range 0–1000 (integers)",
        "[45, 120, 400, 600]",
    ),
    "xyxy_pixel": (
        "[x1, y1, x2, y2]",
        "in absolute pixel coordinates (integers)",
        "[45, 120, 400, 600]",
    ),
}


def _think_prompt(box_format: str) -> str:
    fmt_label, fmt_norm_desc, ex = _BOX_FORMAT_DESC[box_format]
    return (
        "You will be shown an image together with a question that requires "
        "reasoning over multiple visual objects.\n\n"
        "Output two sections, in this order:\n"
        f"1) <think>...</think> — a step-by-step reasoning trace. Every object "
        f"you reference as visual evidence MUST be followed inline by its "
        f"bounding box written as {fmt_label} {fmt_norm_desc}. "
        "List ALL relevant evidence objects (usually 2–4), not only the answer.\n"
        "2) <answer>...</answer> — a concise sentence that directly answers the "
        "question. Include the bounding box ONLY for the object(s) that are the "
        "direct answer (usually 1). Do NOT repeat the boxes of the supporting "
        "evidence here.\n\n"
        "Format:\n"
        f"<think>...the boat {ex} is moored next to the bridge {ex}...</think>\n"
        f"<answer>The boat {ex} is the answer.</answer>\n\n"
        "Strict rules:\n"
        f"- Use only the bracket format {fmt_label}; do NOT emit JSON code "
        "blocks, markdown, or `box_2d`/`bbox_2d` keys.\n"
        "- Do NOT write anything outside the <think> and <answer> tags.\n"
        "- Output coordinates as integers.\n\n"
        "Question: {question}"
    )


def _no_think_prompt(box_format: str) -> str:
    fmt_label, fmt_norm_desc, _ = _BOX_FORMAT_DESC[box_format]
    return (
        "Look at the image and answer the question. Wrap the answer in "
        "<answer> </answer> tags. For every object you mention as part of the "
        f"answer, append its bounding box inline as {fmt_label} {fmt_norm_desc}. "
        "Do not emit JSON code blocks or anything outside the <answer> tags.\n\n"
        "Question: {question}"
    )


def build_prompt(question: str, use_thinking: bool, box_format: str = "yxyx_norm1000") -> str:
    template = _think_prompt(box_format) if use_thinking else _no_think_prompt(box_format)
    return template.format(question=question)


# ───────────────────────── OpenRouter client ─────────────────────────────

class OpenRouterClient:
    """Thin wrapper around the OpenAI SDK pointed at OpenRouter."""

    def __init__(self, api_key: str, model: str,
                 base_url: str = "https://openrouter.ai/api/v1",
                 referer: Optional[str] = None,
                 site_title: Optional[str] = None,
                 max_tokens: int = 4096,
                 temperature: float = 0.0,
                 timeout: float = 120.0):
        import openai
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        extra_headers = {}
        if referer:
            extra_headers["HTTP-Referer"] = referer
        if site_title:
            extra_headers["X-Title"] = site_title
        self._extra_headers = extra_headers or None
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def query(self, image: Image.Image, prompt: str, max_retries: int = 4) -> str:
        b64 = encode_pil_to_base64(image)
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }]
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    extra_headers=self._extra_headers,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last_err = e
                wait = min(2 ** attempt, 30) + 0.5 * attempt
                print(f"[openrouter] attempt {attempt + 1}/{max_retries} failed: {e!r} — sleeping {wait:.1f}s")
                time.sleep(wait)
        print(f"[openrouter] giving up after {max_retries} attempts: {last_err!r}")
        return ""


# ───────────────────────────── main ──────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate any vision LLM on VER via OpenRouter.")
    p.add_argument("--packed_tfrecord",
                   default=os.path.join(PROJECT_ROOT, "data/VRT-Eval/vrt_eval.tfrecord"))
    p.add_argument("--env_file", default=os.path.join(PROJECT_ROOT, ".env"))
    p.add_argument("--model", default="google/gemini-2.5-pro",
                   help="OpenRouter model id (e.g. google/gemini-3.1-pro-preview, "
                        "qwen/qwen3-vl-8b-instruct, openai/gpt-5.4).")
    p.add_argument("--base_url", default="https://openrouter.ai/api/v1")
    p.add_argument("--output_dir", default="work_dirs/openrouter_eval")
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--no-thinking", dest="use_thinking", action="store_false")
    p.add_argument("--box_format", default=None, choices=list(BOX_FORMATS) + [None],
                   help="Default: yxyx_norm1000 for google/*, xyxy_norm1000 for qwen/*.")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--referer", default="https://github.com/Sa2VA")
    p.add_argument("--site_title", default="Sa2VA-VER-Eval")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--visualize", action="store_true")
    p.add_argument("--visualize-low-iou-only", action="store_true")
    p.add_argument("--sam2_checkpoint",
                   default=os.path.join(PROJECT_ROOT, "pretrained/sam2/sam2_hiera_large.pt"))
    p.add_argument("--sam2_config", default="sam2_hiera_l.yaml")
    p.add_argument("--sam2_device", default=None)
    p.add_argument("--skip_sam2", action="store_true")
    return p.parse_args()


def predict_sample(sample: Dict, client: OpenRouterClient, use_thinking: bool,
                   box_format: str) -> Dict:
    """Phase-1: build strict prompt, query model, build record."""
    prompt = build_prompt(sample["question"], use_thinking, box_format)
    pred_text = client.query(sample["image"], prompt)
    return build_record(sample, pred_text, box_format)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.box_format is None:
        args.box_format = auto_box_format_for_model(args.model)
    print(f"[cfg] model={args.model}  box_format={args.box_format}")

    load_env_from_file(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit(
            f"[error] OPENROUTER_API_KEY not found. Add it to {args.env_file} "
            "or export it in the shell before running."
        )

    print(f"[data] loading packed eval tfrecord: {args.packed_tfrecord}")
    dataset = PackedVRTEvalDataset(args.packed_tfrecord)
    keys = sorted(dataset.get_all_keys())
    if args.max_samples:
        keys = keys[: args.max_samples]
    print(f"[data] {len(keys)} samples to evaluate")

    # Resume: accept the current or the older bbox-only filename.
    results_path = os.path.join(args.output_dir, "ver_results.json")
    summary_path = os.path.join(args.output_dir, "ver_summary.json")
    older_path = os.path.join(args.output_dir, "ver_bbox_results.json")
    done_keys: set = set()
    existing_results: List[Dict] = []
    if args.resume:
        load_from = results_path if os.path.isfile(results_path) else (
            older_path if os.path.isfile(older_path) else None)
        if load_from:
            try:
                with open(load_from, "r") as f:
                    existing_results = json.load(f)
                done_keys = {r["sample_info"]["key"] for r in existing_results}
                print(f"[resume] loaded {len(done_keys)} samples from {load_from}")
            except Exception as e:  # noqa: BLE001
                print(f"[resume] failed to load existing results, starting fresh: {e!r}")
    pending_keys = [k for k in keys if k not in done_keys]

    client = OpenRouterClient(
        api_key=api_key, model=args.model, base_url=args.base_url,
        referer=args.referer, site_title=args.site_title,
        max_tokens=args.max_tokens, temperature=args.temperature,
    )

    # ── Phase 1: parallel HTTP inference ─────────────────────────────────
    results: List[Dict] = list(existing_results)
    write_lock = threading.Lock()

    def _predict(key: str) -> Optional[Dict]:
        sample = dataset.get_sample(key)
        if not sample:
            print(f"[warn] could not load sample {key}")
            return None
        try:
            return predict_sample(sample, client, args.use_thinking, args.box_format)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] sample {key} failed: {e!r}")
            return None

    def _checkpoint():
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    if args.workers <= 1:
        for key in tqdm.tqdm(pending_keys, desc="openrouter"):
            res = _predict(key)
            if res is not None:
                results.append(res)
                if len(results) % 10 == 0:
                    with write_lock:
                        _checkpoint()
            if args.sleep > 0:
                time.sleep(args.sleep)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_predict, k): k for k in pending_keys}
            for fut in tqdm.tqdm(as_completed(futures), total=len(futures), desc="openrouter"):
                res = fut.result()
                if res is not None:
                    with write_lock:
                        results.append(res)
                        if len(results) % 10 == 0:
                            _checkpoint()
    _checkpoint()
    print(f"[openrouter] phase complete: {len(results)} records → {results_path}")

    # ── Phase 2: SAM2 → mask IoU (sequential GPU) ────────────────────────
    if args.skip_sam2:
        print("[sam2] skipped (--skip_sam2)")
    else:
        todo = [r for r in results
                if not r["evaluation"]["reasoning_ious"]
                and not r["evaluation"]["answer_ious"]
                and (r["prediction"]["reasoning_bboxes"] or r["prediction"]["answer_bboxes"]
                     or r["ground_truth"]["reasoning_masks_rle"]
                     or r["ground_truth"]["answer_masks_rle"])]
        if todo:
            sam2 = SAM2BoxToMask(args.sam2_checkpoint, args.sam2_config, args.sam2_device)
            for rec in tqdm.tqdm(todo, desc="sam2"):
                try:
                    score_with_sam2(rec, dataset, sam2)
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] SAM2 failed for {rec['sample_info']['key']}: {e!r}")

    _checkpoint()
    print(f"[out] detailed results → {results_path}")

    summary = aggregate_metrics(results, model_label=args.model)
    summary.update({
        "model": args.model,
        "base_url": args.base_url,
        "use_thinking": args.use_thinking,
        "box_format": args.box_format,
        "prompt_style": "strict",
        "total_samples": len(results),
        "detailed_results": results,
    })
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[out] summary → {summary_path}")
    print_summary(summary)

    if args.visualize:
        from projects.vrt_sa2va.eval_openrouter.visualize_results import visualize
        visualize(args.output_dir, args.packed_tfrecord,
                  low_iou_only=args.visualize_low_iou_only)


if __name__ == "__main__":
    main()
