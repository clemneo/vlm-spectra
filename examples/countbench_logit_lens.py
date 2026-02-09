"""CountBench Logit Lens Number Token Experiment.

Systematically investigates whether number tokens appear in the logit lens
of visual tokens when a VLM processes counting images from CountBench.

For each image, we:
1. Generate the model's counting answer
2. Run a forward pass caching all residual stream activations
3. Compute logit lens on image tokens only
4. Analyze which number tokens appear and at what layers/positions

Usage:
    uv run python examples/countbench_logit_lens.py --num-images 5
    uv run python examples/countbench_logit_lens.py --num-images 100 --top-k 10
    uv run python examples/countbench_logit_lens.py --num-images -1  # all images
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import traceback
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

from vlm_spectra import ActivationCache, HookedVLM
from vlm_spectra.analysis.logit_lens import compute_logit_lens

# ---------------------------------------------------------------------------
# Number token utilities
# ---------------------------------------------------------------------------

WORD_TO_NUMBER = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

# Build reverse mapping for parsing generated text (longest match first)
NUMBER_WORDS_BY_LENGTH = sorted(WORD_TO_NUMBER.keys(), key=len, reverse=True)


def token_to_number(token_str: str) -> int | None:
    """Map a decoded token string to an integer value, or None."""
    stripped = token_str.strip().lower()
    # Digit strings "0" through "20"
    if stripped.isdecimal():
        val = int(stripped)
        if 0 <= val <= 20:
            return val
        return None
    # Word forms
    return WORD_TO_NUMBER.get(stripped)


def parse_count_from_text(text: str) -> int | None:
    """Extract an integer count from generated text.

    Tries standalone digits first, then word forms (longest match first).
    """
    # Try regex for standalone digit(s)
    match = re.search(r"\b(\d+)\b", text)
    if match:
        return int(match.group(1))
    # Fall back to word-form matching
    text_lower = text.lower()
    for word in NUMBER_WORDS_BY_LENGTH:
        if re.search(r"\b" + word + r"\b", text_lower):
            return WORD_TO_NUMBER[word]
    return None


# ---------------------------------------------------------------------------
# CountBench item parsing
# ---------------------------------------------------------------------------


def parse_countbench_item(item: dict) -> tuple[int, str, str | None]:
    """Parse a CountBench dataset item.

    Returns (ground_truth_count, caption, object_name_or_None).
    CountBench captions typically have the form "a photo of {number} {objects}".
    """
    caption = item.get("text", item.get("caption", ""))
    number = item.get("number", item.get("count", None))

    if number is not None:
        ground_truth = int(number)
    else:
        parsed = parse_count_from_text(caption)
        ground_truth = parsed if parsed is not None else -1

    # Try to extract object name from caption
    object_name = None
    # Pattern: "... {number} {object_description}"
    match = re.search(
        r"\b(?:of\s+)?(\d+|"
        + "|".join(NUMBER_WORDS_BY_LENGTH)
        + r")\s+(.+?)(?:\s+on\b|\s+in\b|\s+at\b|\s+with\b|\.|$)",
        caption.lower(),
    )
    if match:
        object_name = match.group(2).strip()
        # Clean up trailing articles or common noise
        object_name = re.sub(r"\s+$", "", object_name)
        if object_name:
            object_name = object_name.rstrip(".")

    return ground_truth, caption, object_name if object_name else None


# ---------------------------------------------------------------------------
# Per-image logit lens analysis
# ---------------------------------------------------------------------------


def analyze_logit_lens_numbers(
    logit_lens_result: list[list[list[tuple[str, str]]]],
    ground_truth: int,
    model_answer: int | None,
) -> dict:
    """Analyze number tokens in logit lens output for one image.

    Args:
        logit_lens_result: [layer][pos][(token_str, prob_str)] from compute_logit_lens
        ground_truth: the correct count
        model_answer: the model's generated count (or None)

    Returns:
        Dictionary with per-layer and aggregate number token statistics.
    """
    num_layers = len(logit_lens_result)
    num_positions = len(logit_lens_result[0]) if num_layers > 0 else 0

    per_layer_stats = []

    for layer_idx in range(num_layers):
        layer_data = logit_lens_result[layer_idx]
        positions_with_number = 0
        positions_with_gt = 0
        positions_with_answer = 0
        number_histogram = Counter()
        top1_number_histogram = Counter()

        for pos_idx, token_prob_pairs in enumerate(layer_data):
            found_any_number = False
            for rank, (tok_str, prob_str) in enumerate(token_prob_pairs):
                num_val = token_to_number(tok_str)
                if num_val is not None:
                    if not found_any_number:
                        positions_with_number += 1
                        found_any_number = True
                    number_histogram[num_val] += 1
                    if rank == 0:
                        top1_number_histogram[num_val] += 1
                    if num_val == ground_truth:
                        positions_with_gt += 1
                    if model_answer is not None and num_val == model_answer:
                        positions_with_answer += 1

        per_layer_stats.append(
            {
                "layer": layer_idx,
                "num_positions": num_positions,
                "positions_with_number": positions_with_number,
                "frac_positions_with_number": (
                    positions_with_number / num_positions if num_positions > 0 else 0
                ),
                "positions_with_gt": positions_with_gt,
                "positions_with_answer": positions_with_answer,
                "number_histogram": dict(number_histogram),
                "top1_number_histogram": dict(top1_number_histogram),
            }
        )

    # Aggregate: any layer has a number token?
    any_number = any(s["positions_with_number"] > 0 for s in per_layer_stats)
    any_gt = any(s["positions_with_gt"] > 0 for s in per_layer_stats)
    any_answer = any(s["positions_with_answer"] > 0 for s in per_layer_stats)

    # Late-layer (last 1/3) aggregate histogram
    late_start = num_layers - max(1, num_layers // 3)
    late_histogram = Counter()
    late_top1_histogram = Counter()
    for s in per_layer_stats[late_start:]:
        late_histogram.update(s["number_histogram"])
        late_top1_histogram.update(s["top1_number_histogram"])

    most_common_late = late_histogram.most_common(1)[0][0] if late_histogram else None
    most_common_late_top1 = (
        late_top1_histogram.most_common(1)[0][0] if late_top1_histogram else None
    )

    return {
        "num_layers": num_layers,
        "num_image_positions": num_positions,
        "per_layer": per_layer_stats,
        "any_number_in_any_layer": any_number,
        "gt_in_any_layer": any_gt,
        "answer_in_any_layer": any_answer,
        "late_layer_number_histogram": dict(late_histogram),
        "late_layer_top1_histogram": dict(late_top1_histogram),
        "most_common_number_late_layers": most_common_late,
        "most_common_number_late_layers_top1": most_common_late_top1,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------


def compute_aggregate_stats(results: list[dict]) -> dict:
    """Compute aggregate statistics over all per-image results."""
    n = len(results)
    if n == 0:
        return {}

    # Model accuracy
    correct = sum(
        1
        for r in results
        if r["model_answer"] is not None and r["model_answer"] == r["ground_truth"]
    )
    model_accuracy = correct / n

    # Number token prevalence
    any_number_count = sum(
        1 for r in results if r["analysis"]["any_number_in_any_layer"]
    )
    gt_count = sum(1 for r in results if r["analysis"]["gt_in_any_layer"])
    answer_count = sum(1 for r in results if r["analysis"]["answer_in_any_layer"])

    # Per-layer breakdown: average fraction of positions with number tokens
    if results and results[0]["analysis"]["per_layer"]:
        num_layers = results[0]["analysis"]["num_layers"]
        per_layer_avg = []
        for layer_idx in range(num_layers):
            fracs = [
                r["analysis"]["per_layer"][layer_idx]["frac_positions_with_number"]
                for r in results
                if layer_idx < len(r["analysis"]["per_layer"])
            ]
            per_layer_avg.append(
                {
                    "layer": layer_idx,
                    "avg_frac_positions_with_number": sum(fracs) / len(fracs)
                    if fracs
                    else 0,
                }
            )
    else:
        num_layers = 0
        per_layer_avg = []

    # Correlation analysis (late-layer most common number vs model answer / ground truth)
    correlation_data = _compute_correlations(results)

    # Top-1 analysis
    top1_any_number = sum(
        1
        for r in results
        if any(s["top1_number_histogram"] for s in r["analysis"]["per_layer"])
    )

    top1_gt_in_any_layer = sum(
        1
        for r in results
        if any(
            str(r["ground_truth"]) in {str(k) for k in s["top1_number_histogram"]}
            for s in r["analysis"]["per_layer"]
        )
    )

    top1_answer_in_any_layer = sum(
        1
        for r in results
        if r["model_answer"] is not None
        and any(
            str(r["model_answer"]) in {str(k) for k in s["top1_number_histogram"]}
            for s in r["analysis"]["per_layer"]
        )
    )

    return {
        "num_images": n,
        "model_accuracy": model_accuracy,
        "number_token_prevalence": any_number_count / n,
        "gt_prevalence": gt_count / n,
        "answer_prevalence": answer_count / n,
        "top1_number_prevalence": top1_any_number / n,
        "top1_gt_prevalence": top1_gt_in_any_layer / n,
        "top1_answer_prevalence": top1_answer_in_any_layer / n,
        "per_layer_avg": per_layer_avg,
        "correlations": correlation_data,
    }


def _compute_correlations(results: list[dict]) -> dict:
    """Compute Spearman correlations between late-layer most common number
    and model answer / ground truth."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return {"note": "scipy not installed, skipping correlation analysis"}

    # Pairs for correlation: (most_common_late, model_answer) and (most_common_late, gt)
    pairs_answer = []
    pairs_gt = []

    for r in results:
        mc = r["analysis"]["most_common_number_late_layers"]
        if mc is None:
            continue
        gt = r["ground_truth"]
        ma = r["model_answer"]
        pairs_gt.append((mc, gt))
        if ma is not None:
            pairs_answer.append((mc, ma))

    corr_data = {}
    if len(pairs_gt) >= 3:
        xs, ys = zip(*pairs_gt)
        rho, pval = spearmanr(xs, ys)
        corr_data["late_most_common_vs_gt"] = {
            "rho": rho,
            "pval": pval,
            "n": len(pairs_gt),
        }

    if len(pairs_answer) >= 3:
        xs, ys = zip(*pairs_answer)
        rho, pval = spearmanr(xs, ys)
        corr_data["late_most_common_vs_answer"] = {
            "rho": rho,
            "pval": pval,
            "n": len(pairs_answer),
        }

    # Top-1 variant
    pairs_gt_top1 = []
    pairs_answer_top1 = []
    for r in results:
        mc = r["analysis"]["most_common_number_late_layers_top1"]
        if mc is None:
            continue
        gt = r["ground_truth"]
        ma = r["model_answer"]
        pairs_gt_top1.append((mc, gt))
        if ma is not None:
            pairs_answer_top1.append((mc, ma))

    if len(pairs_gt_top1) >= 3:
        xs, ys = zip(*pairs_gt_top1)
        rho, pval = spearmanr(xs, ys)
        corr_data["late_top1_vs_gt"] = {
            "rho": rho,
            "pval": pval,
            "n": len(pairs_gt_top1),
        }

    if len(pairs_answer_top1) >= 3:
        xs, ys = zip(*pairs_answer_top1)
        rho, pval = spearmanr(xs, ys)
        corr_data["late_top1_vs_answer"] = {
            "rho": rho,
            "pval": pval,
            "n": len(pairs_answer_top1),
        }

    return corr_data


def print_summary(stats: dict) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 70)
    print("AGGREGATE STATISTICS")
    print("=" * 70)
    print(f"  Images processed:            {stats['num_images']}")
    print(f"  Model counting accuracy:     {stats['model_accuracy']:.3f}")
    print()
    print("  --- Number Token Prevalence (any top-k) ---")
    print(f"  Any number in logit lens:    {stats['number_token_prevalence']:.3f}")
    print(f"  Ground truth in logit lens:  {stats['gt_prevalence']:.3f}")
    print(f"  Model answer in logit lens:  {stats['answer_prevalence']:.3f}")
    print()
    print("  --- Number Token Prevalence (top-1 only) ---")
    print(f"  Any number (top-1):          {stats['top1_number_prevalence']:.3f}")
    print(f"  Ground truth (top-1):        {stats['top1_gt_prevalence']:.3f}")
    print(f"  Model answer (top-1):        {stats['top1_answer_prevalence']:.3f}")
    print()

    if stats.get("correlations"):
        print("  --- Correlations (Spearman) ---")
        for key, val in stats["correlations"].items():
            if isinstance(val, dict) and "rho" in val:
                print(
                    f"  {key}: rho={val['rho']:.3f}, p={val['pval']:.4f} (n={val['n']})"
                )
            else:
                print(f"  {key}: {val}")
        print()

    # Per-layer summary (sample a few layers)
    if stats.get("per_layer_avg"):
        layers = stats["per_layer_avg"]
        num_layers = len(layers)
        # Show every ~8th layer + last layer
        step = max(1, num_layers // 8)
        sample_idxs = list(range(0, num_layers, step))
        if (num_layers - 1) not in sample_idxs:
            sample_idxs.append(num_layers - 1)
        print("  --- Per-Layer Avg Fraction of Image Positions with Number Token ---")
        for idx in sample_idxs:
            layer_info = layers[idx]
            print(
                f"  Layer {layer_info['layer']:3d}: "
                f"{layer_info['avg_frac_positions_with_number']:.4f}"
            )
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="CountBench Logit Lens Number Token Experiment"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nielsr/countbench",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to process (-1 for all)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-k tokens to consider in logit lens",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/countbench_logit_lens",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max tokens for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Setup
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    per_image_dir = output_dir / "per_image"
    per_image_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset: {args.dataset}")
    from datasets import load_dataset

    ds = load_dataset(args.dataset, split="train")
    total_images = len(ds)
    num_images = (
        total_images if args.num_images == -1 else min(args.num_images, total_images)
    )
    print(f"Dataset loaded: {total_images} images, processing {num_images}")

    # Load model (default_prompt=None to avoid tool-calling template)
    print(f"Loading model: {args.model}")
    model = HookedVLM.from_pretrained(args.model, default_prompt=None)
    print(f"Model loaded: {model.lm_num_layers} layers, device={model.device}")

    components = model.get_model_components()
    norm = components["norm"]
    lm_head = components["lm_head"]
    tokenizer = components["tokenizer"]

    # Process images
    results = []
    errors = []

    for idx in tqdm(range(num_images), desc="Processing images"):
        result_path = per_image_dir / f"image_{idx:05d}.json"

        # Resume support: skip if already processed
        if result_path.exists():
            with open(result_path) as f:
                result = json.load(f)
            results.append(result)
            continue

        try:
            item = ds[idx]
            ground_truth, caption, object_name = parse_countbench_item(item)

            # Build prompt
            if object_name:
                prompt = f"How many {object_name} are in this image? Answer with just the number."
            else:
                prompt = (
                    "How many objects are in this image? Answer with just the number."
                )

            image = item["image"]

            # Step 1: Generate model answer
            inputs = model.prepare_messages(prompt, image)
            outputs = model.generate(
                inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
            generated_ids = outputs.sequences[0]
            input_len = inputs["input_ids"].shape[-1]
            generated_text = tokenizer.decode(
                generated_ids[input_len:], skip_special_tokens=True
            )
            model_answer = parse_count_from_text(generated_text)

            # Step 2: Get image token range
            start_idx, end_idx = model.get_image_token_range(inputs)

            # Step 3: Cache hidden states and forward
            with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
                model.forward(inputs)

            # Step 4: Stack and slice to image tokens only
            activation_cache = ActivationCache()
            activation_cache._data = model.cache
            stacked = activation_cache.stack("lm.blocks.*.hook_resid_post")
            # stacked shape: [num_layers, batch, seq, hidden]
            image_hidden = stacked[:, 0, start_idx : end_idx + 1, :]
            # image_hidden shape: [num_layers, num_image_tokens, hidden]

            # Step 5: Compute logit lens on image tokens
            logit_lens_result = compute_logit_lens(
                image_hidden, norm, lm_head, tokenizer, top_k=args.top_k
            )

            # Step 6: Analyze number tokens
            analysis = analyze_logit_lens_numbers(
                logit_lens_result, ground_truth, model_answer
            )

            result = {
                "idx": idx,
                "caption": caption,
                "ground_truth": ground_truth,
                "object_name": object_name,
                "prompt": prompt,
                "generated_text": generated_text,
                "model_answer": model_answer,
                "num_image_tokens": end_idx - start_idx + 1,
                "analysis": analysis,
            }

            # Save per-image result
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)

            results.append(result)

        except Exception as e:
            error_info = {
                "idx": idx,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            errors.append(error_info)
            print(f"\nError on image {idx}: {e}", file=sys.stderr)
            continue
        finally:
            # Cleanup to free GPU memory
            model.cache = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compute and save aggregate statistics
    if results:
        stats = compute_aggregate_stats(results)
        stats_path = output_dir / "aggregate_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"\nAggregate stats saved to {stats_path}")
        print_summary(stats)
    else:
        print("No results to aggregate.")

    if errors:
        errors_path = output_dir / "errors.json"
        with open(errors_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"\n{len(errors)} errors saved to {errors_path}")


if __name__ == "__main__":
    main()
