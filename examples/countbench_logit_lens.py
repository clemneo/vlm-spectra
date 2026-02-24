"""CountBench Logit Lens Number Token Experiment.

Systematically investigates whether number tokens appear in the logit lens
of visual tokens when a VLM processes counting images from CountBench.

For each image, we:
1. Generate the model's counting answer
2. Run a forward pass caching all residual stream activations
3. Compute logit lens on image tokens only
4. Check if the model's generated count matches the most frequent number
   in the top-1 logit lens predictions (rank-based metric)

Usage:
    uv run python examples/countbench_logit_lens.py --num-images 5
    uv run python examples/countbench_logit_lens.py --num-images 100 --top-k 10
    uv run python examples/countbench_logit_lens.py --num-images -1  # all images
    uv run python examples/countbench_logit_lens.py --analyze-only --prob-threshold 0.1
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import re
import sys
import traceback
from pathlib import Path

import torch
from tqdm import tqdm

from vlm_spectra import ActivationCache, HookedVLM
from vlm_spectra.analysis.logit_lens import compute_logit_lens

# ---------------------------------------------------------------------------
# Number token utilities
# ---------------------------------------------------------------------------

WORD_TO_NUMBER = {
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}

# Build reverse mapping for parsing generated text (longest match first)
NUMBER_WORDS_BY_LENGTH = sorted(WORD_TO_NUMBER.keys(), key=len, reverse=True)


def token_to_number(token_str: str) -> int | None:
    """Map a decoded token string to an integer value, or None."""
    stripped = token_str.strip().lower()
    # Digit strings "2" through "9"
    if stripped.isdecimal():
        val = int(stripped)
        if 2 <= val <= 9:
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
# Rank-based logit lens analysis
# ---------------------------------------------------------------------------


def compute_top1_number_counts(
    logit_lens_result: list[list[list[tuple[str, str]]]],
    prob_threshold: float = 0.0,
) -> dict[int, list[float]]:
    """Count how often each number appears as the top-1 logit lens prediction.

    For each (layer, position), checks if the top-1 token maps to a number
    via token_to_number(). If yes and prob >= prob_threshold, appends the
    probability to that number's list.

    Returns:
        e.g. {5: [0.12, 0.08, 0.15], 3: [0.09]}
    """
    counts: dict[int, list[float]] = {}
    for layer_data in logit_lens_result:
        for token_prob_pairs in layer_data:
            if not token_prob_pairs:
                continue
            tok_str, prob_str = token_prob_pairs[0]  # top-1
            prob = float(prob_str)
            if prob < prob_threshold:
                continue
            num_val = token_to_number(tok_str)
            if num_val is not None:
                counts.setdefault(num_val, []).append(prob)
    return counts


def compute_answer_rank(
    number_counts: dict[int, list[float]], model_answer: int | None
) -> int | None:
    """Compute competition rank of model_answer among number counts.

    Numbers are sorted by count (len of probs list), descending.
    Competition ranking: rank = 1 + count of numbers with strictly higher frequency.

    Returns None if model_answer is None or absent from counts.
    """
    if model_answer is None or model_answer not in number_counts:
        return None
    answer_freq = len(number_counts[model_answer])
    rank = 1 + sum(
        1 for n, probs in number_counts.items() if len(probs) > answer_freq
    )
    return rank


def recompute_cached_ranks(result: dict, prob_threshold: float) -> None:
    """Recompute cached ranks for a new probability threshold."""
    stored_counts = result.get("number_counts", {})
    if not stored_counts:
        return

    filtered_counts: dict[int, list[float]] = {}
    for num_str, probs in stored_counts.items():
        filtered = [p for p in probs if p >= prob_threshold]
        if filtered:
            filtered_counts[int(num_str)] = filtered

    unfiltered_counts = {int(k): v for k, v in stored_counts.items()}
    model_answer = result.get("model_answer")
    result["rank_unfiltered"] = compute_answer_rank(
        unfiltered_counts, model_answer
    )
    result["rank_filtered"] = compute_answer_rank(filtered_counts, model_answer)
    result["prob_threshold"] = prob_threshold


def create_rank_histogram(
    ranks: list[int],
    title: str,
    output_path: Path,
    max_rank_display: int = 15,
) -> None:
    """Create a bar chart of answer ranks.

    Bins 1..max_rank_display plus an overflow ">N" bin.
    Rank 1 bar is highlighted in coral, rest in steelblue.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Bin ranks
    bin_counts = [0] * (max_rank_display + 1)  # index 0 unused, 1..max_rank_display + overflow
    for r in ranks:
        if r <= max_rank_display:
            bin_counts[r] += 1
        else:
            bin_counts[0] += 1  # overflow stored at index 0 temporarily

    overflow = bin_counts[0]
    bar_counts = bin_counts[1:] + [overflow]
    labels = [str(i) for i in range(1, max_rank_display + 1)] + [f">{max_rank_display}"]
    colors = ["coral" if i == 0 else "steelblue" for i in range(len(bar_counts))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(bar_counts)), bar_counts, color=colors)

    # Annotate bars with counts
    for bar, count in zip(bars, bar_counts):
        if count > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                str(count),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Rank of model answer among top-1 number counts")
    ax.set_ylabel("Number of images")
    ax.set_title(title)

    # Add n= and rank-1 % annotation
    n = len(ranks)
    rank1_count = bar_counts[0] if bar_counts else 0
    rank1_pct = rank1_count / n * 100 if n > 0 else 0
    ax.text(
        0.95,
        0.95,
        f"n={n}\nrank-1: {rank1_pct:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Histogram saved to {output_path}")


def print_rank_summary(
    ranks_unfiltered: list[int],
    ranks_filtered: list[int],
    num_images: int,
    num_excluded_unfiltered: int,
    num_excluded_filtered: int,
) -> None:
    """Print a formatted summary table of rank statistics."""
    print("\n" + "=" * 70)
    print("RANK SUMMARY")
    print("=" * 70)
    print(f"  Total images processed:      {num_images}")
    print()

    for label, ranks, num_excluded in [
        ("Unfiltered (threshold=0)", ranks_unfiltered, num_excluded_unfiltered),
        ("Filtered", ranks_filtered, num_excluded_filtered),
    ]:
        n = len(ranks)
        if n == 0:
            print(f"  --- {label} ---")
            print(f"  Computable ranks:            0 / {num_images}")
            print(f"  Excluded (no rank):          {num_excluded}")
            print()
            continue

        sorted_ranks = sorted(ranks)
        median_rank = sorted_ranks[n // 2] if n % 2 == 1 else (
            (sorted_ranks[n // 2 - 1] + sorted_ranks[n // 2]) / 2
        )
        mean_rank = sum(ranks) / n
        rank1_count = sum(1 for r in ranks if r == 1)

        print(f"  --- {label} ---")
        print(f"  Computable ranks:            {n} / {num_images}")
        print(f"  Excluded (no rank):          {num_excluded}")
        print(f"  Median rank:                 {median_rank:.1f}")
        print(f"  Mean rank:                   {mean_rank:.2f}")
        print(f"  Fraction at rank 1:          {rank1_count}/{n} ({rank1_count/n:.3f})")
        print()

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
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip forward passes, reanalyze from cached per-image results",
    )
    parser.add_argument(
        "--prob-threshold",
        type=float,
        default=0.05,
        help="Min probability for filtered histogram (default: 0.05)",
    )
    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    per_image_dir = output_dir / "per_image"
    cache_dir = output_dir / "cache"
    plots_dir = output_dir / "plots"

    if args.analyze_only:
        # --analyze-only mode: reanalyze from cached per-image results
        result_files = sorted(glob.glob(str(per_image_dir / "image_*.json")))
        if not result_files:
            print(f"No cached results found in {per_image_dir}")
            return

        print(f"Analyze-only mode: found {len(result_files)} cached results")
        print(f"Re-filtering with prob_threshold={args.prob_threshold}")

        results = []
        for result_path in tqdm(result_files, desc="Reanalyzing"):
            with open(result_path) as f:
                result = json.load(f)

            # Re-filter number_counts at new threshold from stored probabilities
            recompute_cached_ranks(result, args.prob_threshold)

            results.append(result)

    else:
        # Normal mode: run forward passes
        torch.manual_seed(args.seed)
        per_image_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Load dataset
        print(f"Loading dataset: {args.dataset}")
        from datasets import load_dataset

        ds = load_dataset(args.dataset, split="train")
        total_images = len(ds)

        # Pre-filter to valid indices: non-null images with ground truth in 2–9
        valid_indices = []
        for i in range(total_images):
            item = ds[i]
            if item["image"] is None:
                continue
            gt, _, _ = parse_countbench_item(item)
            if 2 <= gt <= 9:
                valid_indices.append(i)
        print(f"Dataset loaded: {total_images} total, {len(valid_indices)} valid (count 2–9, non-null)")

        if args.num_images == -1:
            selected_indices = valid_indices
        else:
            selected_indices = valid_indices[: args.num_images]
        num_images = len(selected_indices)
        print(f"Processing {num_images} images")

        print(f"Loading model: {args.model}")
        model = HookedVLM.from_pretrained(args.model)
        print(f"Model loaded: {model.lm_num_layers} layers, device={model.device}")

        components = model.get_model_components()
        norm = components["norm"]
        lm_head = components["lm_head"]
        tokenizer = components["tokenizer"]

        # Process images
        results = []
        errors = []

        for idx in tqdm(selected_indices, desc="Processing images"):
            result_path = per_image_dir / f"image_{idx:05d}.json"

            # Resume support: skip if already processed
            if result_path.exists():
                with open(result_path) as f:
                    result = json.load(f)
                recompute_cached_ranks(result, args.prob_threshold)
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

                # Step 6: Save full logit lens cache
                cache_path = cache_dir / f"image_{idx:05d}_logit_lens.json"
                with open(cache_path, "w") as f:
                    json.dump(logit_lens_result, f)

                # Step 7: Compute rank-based metrics
                number_counts = compute_top1_number_counts(logit_lens_result)
                number_counts_filtered = compute_top1_number_counts(
                    logit_lens_result, prob_threshold=args.prob_threshold
                )
                rank_unfiltered = compute_answer_rank(number_counts, model_answer)
                rank_filtered = compute_answer_rank(
                    number_counts_filtered, model_answer
                )

                # Serialize number_counts: keys must be strings for JSON
                number_counts_serializable = {
                    str(k): v for k, v in number_counts.items()
                }

                result = {
                    "idx": idx,
                    "caption": caption,
                    "ground_truth": ground_truth,
                    "object_name": object_name,
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "model_answer": model_answer,
                    "num_image_tokens": end_idx - start_idx + 1,
                    "number_counts": number_counts_serializable,
                    "rank_unfiltered": rank_unfiltered,
                    "rank_filtered": rank_filtered,
                    "prob_threshold": args.prob_threshold,
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

        if errors:
            errors_path = output_dir / "errors.json"
            with open(errors_path, "w") as f:
                json.dump(errors, f, indent=2)
            print(f"\n{len(errors)} errors saved to {errors_path}")

    # Post-loop: aggregate ranks, print summary, create histograms
    if not results:
        print("No results to aggregate.")
        return

    ranks_unfiltered = [
        r["rank_unfiltered"] for r in results if r.get("rank_unfiltered") is not None
    ]
    ranks_filtered = [
        r["rank_filtered"] for r in results if r.get("rank_filtered") is not None
    ]
    num_excluded_unfiltered = len(results) - len(ranks_unfiltered)
    num_excluded_filtered = len(results) - len(ranks_filtered)

    # Save aggregate rank results
    plots_dir.mkdir(parents=True, exist_ok=True)
    rank_results = {
        "num_images": len(results),
        "prob_threshold": args.prob_threshold,
        "ranks_unfiltered": ranks_unfiltered,
        "ranks_filtered": ranks_filtered,
        "num_excluded_unfiltered": num_excluded_unfiltered,
        "num_excluded_filtered": num_excluded_filtered,
    }
    rank_results_path = output_dir / "rank_results.json"
    with open(rank_results_path, "w") as f:
        json.dump(rank_results, f, indent=2)
    print(f"\nRank results saved to {rank_results_path}")

    # Print summary
    print_rank_summary(
        ranks_unfiltered,
        ranks_filtered,
        len(results),
        num_excluded_unfiltered,
        num_excluded_filtered,
    )

    # Create histograms
    if ranks_unfiltered:
        create_rank_histogram(
            ranks_unfiltered,
            "Rank of model answer (unfiltered, threshold=0)",
            plots_dir / "rank_histogram_unfiltered.png",
        )
    if ranks_filtered:
        create_rank_histogram(
            ranks_filtered,
            f"Rank of model answer (filtered, threshold={args.prob_threshold})",
            plots_dir / "rank_histogram_filtered.png",
        )


if __name__ == "__main__":
    main()
