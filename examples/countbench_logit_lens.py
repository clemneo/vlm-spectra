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

from vlm_spectra import ActivationCache, BlockAttention, HookedVLM
from vlm_spectra.analysis.logit_lens import compute_logit_lens
from vlm_spectra.preprocessing.utils.vision_info import resolve_patch_params, smart_resize

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

# Activation patching pair: children, source (count=2) vs target (count=7)
PATCHING_PAIR = {
    "obj": "children",
    "source_idx": 438,
    "source_count": 2,
    "target_idx": 463,
    "target_count": 7,
}


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


def compute_register_token_analysis(
    embeddings: torch.Tensor,
    logit_lens_result: list[list[list[tuple[str, str]]]],
) -> dict:
    """Identify register tokens and number-prediction positions among image tokens.

    Args:
        embeddings: [num_image_tokens, hidden_dim] — layer 0 input embeddings
            for image tokens only.
        logit_lens_result: [layers][positions][top_k][(token_str, prob_str)]
            from compute_logit_lens.

    Returns:
        Dict with register_positions, number_positions, and norm statistics.
    """
    # Register tokens: positions where L2 norm > mean + 2*std
    norms = torch.norm(embeddings, dim=-1)
    mean_norm = norms.mean().item()
    std_norm = norms.std().item()
    threshold = mean_norm + 2 * std_norm
    register_positions = (norms > threshold).nonzero(as_tuple=True)[0].tolist()

    # Number positions: any position where top-1 prediction is a number at any layer
    number_positions: set[int] = set()
    for layer_data in logit_lens_result:
        for pos, token_prob_pairs in enumerate(layer_data):
            if not token_prob_pairs:
                continue
            tok_str, _ = token_prob_pairs[0]
            if token_to_number(tok_str) is not None:
                number_positions.add(pos)

    return {
        "register_positions": register_positions,
        "number_positions": sorted(number_positions),
        "norm_mean": round(mean_norm, 4),
        "norm_std": round(std_norm, 4),
        "norm_threshold": round(threshold, 4),
    }


def compute_ablation_analysis(
    model, inputs, start_idx, end_idx, number_positions,
    norm, lm_head, tokenizer, top_k,
):
    """Run ablated forward pass and count where numbers appear at non-ablated positions.

    Zeros out `number_positions` (image-token-relative) at layer 0 resid_pre,
    runs a full forward pass, computes logit lens on image tokens, and counts
    positions where a number 2-9 is top-1 (excluding ablated positions).
    """
    if not number_positions:
        return {
            "positions_before": [],
            "num_before": 0,
            "positions_after": [],
            "num_after": 0,
        }

    ablated_set = set(number_positions)

    # Build ablation hook — zeros specific image-token positions at layer 0
    def ablate_hook(module, args, kwargs, output):
        for pos in ablated_set:
            output[0, start_idx + pos, :] = 0
        return output

    # Second forward pass: ablation + cache
    with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
        with model.run_with_hooks([("lm.blocks.0.hook_resid_pre", ablate_hook)]):
            model.forward(inputs)

    abl_cache = ActivationCache()
    abl_cache._data = model.cache
    abl_stacked = abl_cache.stack("lm.blocks.*.hook_resid_post")
    abl_image_hidden = abl_stacked[:, 0, start_idx : end_idx + 1, :]

    abl_logit_lens = compute_logit_lens(
        abl_image_hidden, norm, lm_head, tokenizer, top_k=top_k
    )

    # Count number positions after ablation, excluding ablated positions
    positions_after = set()
    for layer_data in abl_logit_lens:
        for pos, token_prob_pairs in enumerate(layer_data):
            if pos in ablated_set:
                continue
            if not token_prob_pairs:
                continue
            tok_str, _ = token_prob_pairs[0]
            if token_to_number(tok_str) is not None:
                positions_after.add(pos)

    return {
        "positions_before": sorted(ablated_set),
        "num_before": len(ablated_set),
        "positions_after": sorted(positions_after),
        "num_after": len(positions_after),
    }


def compute_attention_blocking(
    model, inputs, start_idx, number_positions,
    original_logits, ground_truth, tokenizer,
):
    """Block attention from last position to count-token positions and measure effect.

    Uses BlockAttention on all layers to set attention mask to -inf from
    the final sequence position to each count-token position (absolute indices).
    Compares argmax prediction and log-prob of correct count token.
    """
    if not number_positions or ground_truth is None or not (2 <= ground_truth <= 9):
        return None

    import torch.nn.functional as F

    seq_len = inputs["input_ids"].shape[-1]
    last_pos = seq_len - 1

    # Convert image-relative positions to absolute
    count_positions_abs = [start_idx + p for p in number_positions]

    # Block attention from last position to count-token positions
    blocking_pairs = [(last_pos, pos) for pos in count_positions_abs]
    blocker = BlockAttention(blocking_pairs, batch_idx=0)

    with model.run_with_hooks([("lm.blocks.*.attn.hook_mask", blocker)]):
        blocked_output = model.forward(inputs)

    blocked_logits = blocked_output.logits[0, -1, :]

    # Get original and blocked predictions
    original_pred_id = original_logits.argmax(dim=-1).item()
    blocked_pred_id = blocked_logits.argmax(dim=-1).item()
    original_pred_token = tokenizer.decode([original_pred_id])
    blocked_pred_token = tokenizer.decode([blocked_pred_id])
    original_pred_count = token_to_number(original_pred_token)
    blocked_pred_count = token_to_number(blocked_pred_token)

    # Log-prob of the correct count token
    gt_token_ids = tokenizer.encode(str(ground_truth), add_special_tokens=False)
    gt_token_id = gt_token_ids[0]

    original_logprobs = F.log_softmax(original_logits.float(), dim=-1)
    blocked_logprobs = F.log_softmax(blocked_logits.float(), dim=-1)

    logprob_before = original_logprobs[gt_token_id].item()
    logprob_after = blocked_logprobs[gt_token_id].item()

    return {
        "num_blocked_positions": len(count_positions_abs),
        "blocked_positions_abs": count_positions_abs,
        "answer_changed": original_pred_id != blocked_pred_id,
        "original_pred_token": original_pred_token,
        "blocked_pred_token": blocked_pred_token,
        "original_pred_count": original_pred_count,
        "blocked_pred_count": blocked_pred_count,
        "logprob_correct_before": round(logprob_before, 4),
        "logprob_correct_after": round(logprob_after, 4),
        "logprob_change": round(logprob_after - logprob_before, 4),
    }


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


def create_aggregate_heatmap(
    results: list[dict],
    output_path: Path,
    resolution: int = 128,
) -> None:
    """Create aggregate spatial heatmap of number token positions.

    For each image, normalizes 1D number_positions to fractional (row, col)
    coordinates using that image's grid_h/grid_w, bins into a common
    resolution x resolution grid, and averages across all images.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    heatmap = np.zeros((resolution, resolution), dtype=np.float64)
    num_contributing = 0

    for r in results:
        reg = r.get("register_token_analysis")
        if reg is None:
            continue
        number_positions = reg.get("number_positions", [])
        grid_h = r.get("grid_h")
        grid_w = r.get("grid_w")
        if not number_positions or grid_h is None or grid_w is None:
            continue

        num_contributing += 1
        for pos in number_positions:
            row_frac = (pos // grid_w + 0.5) / grid_h
            col_frac = (pos % grid_w + 0.5) / grid_w
            bin_r = min(int(row_frac * resolution), resolution - 1)
            bin_c = min(int(col_frac * resolution), resolution - 1)
            heatmap[bin_r, bin_c] += 1

    if num_contributing == 0:
        print("  No images with grid dimensions and number positions — skipping heatmap")
        return

    heatmap /= num_contributing

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap, cmap="hot", interpolation="nearest", origin="upper")
    ax.set_title(f"Aggregate number-token spatial heatmap (n={num_contributing})")
    ax.set_xlabel("Normalized column position")
    ax.set_ylabel("Normalized row position")

    tick_positions = np.linspace(0, resolution - 1, 5)
    tick_labels = [f"{x:.1f}" for x in np.linspace(0, 1, 5)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels)

    fig.colorbar(im, ax=ax, label="Mean count per image")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Heatmap saved to {output_path}")


# ---------------------------------------------------------------------------
# Activation patching
# ---------------------------------------------------------------------------


def compute_activation_patching(
    model, ds, pair, resize_dim, tokenizer,
) -> dict:
    """Run activation patching experiment between a source-target image pair.

    Patches source activations into the target forward pass at each layer,
    separately for image-only and text-only token positions. Measures
    normalized logit difference and whether the argmax prediction flips
    to the source count.

    Normalization: (patched_ld - clean_ld) / (source_ld - clean_ld)
    where ld = logit(source_count) - logit(target_count).
    0 = no effect (patched ≈ clean target), 1 = full recovery (patched ≈ source).
    """
    from PIL import Image as PILImage

    source_idx = pair["source_idx"]
    target_idx = pair["target_idx"]
    source_count = pair["source_count"]
    target_count = pair["target_count"]
    obj_name = pair["obj"]

    # Get and resize images
    source_image = ds[source_idx]["image"].resize(
        (resize_dim, resize_dim), PILImage.LANCZOS
    )
    target_image = ds[target_idx]["image"].resize(
        (resize_dim, resize_dim), PILImage.LANCZOS
    )

    # Build prompt (matching parse_countbench_item pattern)
    prompt = f"How many {obj_name} are in this image? Answer with just the number."

    # Prepare inputs
    source_inputs = model.prepare_messages(prompt, source_image)
    target_inputs = model.prepare_messages(prompt, target_image)

    # Verify image token ranges match
    source_start, source_end = model.get_image_token_range(source_inputs)
    target_start, target_end = model.get_image_token_range(target_inputs)

    source_n_img = source_end - source_start + 1
    target_n_img = target_end - target_start + 1

    print(f"  Source: idx={source_idx}, count={source_count}, "
          f"image_tokens={source_n_img}, range=[{source_start}, {source_end}]")
    print(f"  Target: idx={target_idx}, count={target_count}, "
          f"image_tokens={target_n_img}, range=[{target_start}, {target_end}]")

    assert source_n_img == target_n_img, (
        f"Image token count mismatch: source={source_n_img}, target={target_n_img}"
    )
    assert source_start == target_start, (
        f"Image token start mismatch: source={source_start}, target={target_start}"
    )

    start_idx = source_start
    end_idx = source_end

    # Get token IDs for the count numbers
    source_tok_id = tokenizer.encode(str(source_count), add_special_tokens=False)[0]
    target_tok_id = tokenizer.encode(str(target_count), add_special_tokens=False)[0]

    print(f"  Source count token: '{tokenizer.decode([source_tok_id])}' (id={source_tok_id})")
    print(f"  Target count token: '{tokenizer.decode([target_tok_id])}' (id={target_tok_id})")

    num_layers = model.lm_num_layers

    # --- Cache source activations ---
    print("  Caching source activations...")
    with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
        source_output = model.forward(source_inputs)

    source_cache = ActivationCache()
    source_cache._data = model.cache
    source_activations = source_cache.stack("lm.blocks.*.hook_resid_post")
    # Shape: [num_layers, 1, seq, hidden]
    source_activations = source_activations.detach().clone()

    source_logits = source_output.logits[0, -1, :].detach().clone()
    source_logit_diff = (
        source_logits[source_tok_id] - source_logits[target_tok_id]
    ).item()

    model.cache = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Cache target activations (clean run) ---
    print("  Caching target activations (clean run)...")
    with model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
        target_output = model.forward(target_inputs)

    clean_logits = target_output.logits[0, -1, :].detach().clone()
    clean_logit_diff = (
        clean_logits[source_tok_id] - clean_logits[target_tok_id]
    ).item()

    # Normalization denominator: source_ld - clean_ld
    denominator = source_logit_diff - clean_logit_diff

    print(f"  Source logit diff (source_tok - target_tok): {source_logit_diff:.4f}")
    print(f"  Clean target logit diff (source_tok - target_tok): {clean_logit_diff:.4f}")
    print(f"  Normalization denominator: {denominator:.4f}")

    model.cache = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    clean_pred_id = clean_logits.argmax().item()
    source_pred_id = source_logits.argmax().item()
    print(f"  Clean target prediction: '{tokenizer.decode([clean_pred_id])}'")
    print(f"  Clean source prediction: '{tokenizer.decode([source_pred_id])}'")

    # --- Patching loop ---
    results_by_condition = {
        "image_only": [], "text_only": [],
        "image_last_16": [], "image_last_32": [],
    }

    for layer_idx in tqdm(range(num_layers), desc="  Patching layers"):
        hook_name = f"lm.blocks.{layer_idx}.hook_resid_post"

        for condition in ["image_only", "text_only", "image_last_16", "image_last_32"]:
            source_acts_layer = source_activations[layer_idx]  # [1, seq, hidden]

            if condition == "image_only":
                def patch_fn(
                    module, args, kwargs, output,
                    _s=source_acts_layer, _si=start_idx, _ei=end_idx,
                ):
                    output[0, _si:_ei + 1, :] = _s[0, _si:_ei + 1, :]
                    return output
            elif condition == "text_only":
                def patch_fn(
                    module, args, kwargs, output,
                    _s=source_acts_layer, _si=start_idx, _ei=end_idx,
                ):
                    output[0, :_si, :] = _s[0, :_si, :]
                    output[0, _ei + 1:, :] = _s[0, _ei + 1:, :]
                    return output
            elif condition == "image_last_16":
                def patch_fn(
                    module, args, kwargs, output,
                    _s=source_acts_layer, _ei=end_idx,
                ):
                    output[0, _ei - 15:_ei + 1, :] = _s[0, _ei - 15:_ei + 1, :]
                    return output
            else:  # image_last_32
                def patch_fn(
                    module, args, kwargs, output,
                    _s=source_acts_layer, _ei=end_idx,
                ):
                    output[0, _ei - 31:_ei + 1, :] = _s[0, _ei - 31:_ei + 1, :]
                    return output

            with model.run_with_hooks([(hook_name, patch_fn)]):
                patched_output = model.forward(target_inputs)

            patched_logits = patched_output.logits[0, -1, :]
            logit_diff = (
                patched_logits[source_tok_id] - patched_logits[target_tok_id]
            ).item()

            if abs(denominator) > 1e-6:
                normalized_diff = (logit_diff - clean_logit_diff) / denominator
            else:
                normalized_diff = 0.0

            pred_id = patched_logits.argmax().item()
            pred_flipped = pred_id == source_tok_id

            results_by_condition[condition].append({
                "layer": layer_idx,
                "logit_diff": round(logit_diff, 4),
                "normalized_diff": round(normalized_diff, 4),
                "pred_token_id": pred_id,
                "pred_token": tokenizer.decode([pred_id]),
                "pred_flipped_to_source": pred_flipped,
            })

        # Clean up between layers
        model.cache = None

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Free source activations
    del source_activations

    return {
        "pair": pair,
        "resize_dim": resize_dim,
        "num_layers": num_layers,
        "start_idx": start_idx,
        "end_idx": end_idx,
        "num_image_tokens": end_idx - start_idx + 1,
        "source_tok_id": source_tok_id,
        "target_tok_id": target_tok_id,
        "source_logit_diff": round(source_logit_diff, 4),
        "clean_logit_diff": round(clean_logit_diff, 4),
        "normalization_denominator": round(denominator, 4),
        "clean_pred_token": tokenizer.decode([clean_pred_id]),
        "source_pred_token": tokenizer.decode([source_pred_id]),
        "image_only": results_by_condition["image_only"],
        "text_only": results_by_condition["text_only"],
        "image_last_16": results_by_condition["image_last_16"],
        "image_last_32": results_by_condition["image_last_32"],
    }


def create_patching_plot(result: dict, output_path: Path) -> None:
    """Create a line plot of activation patching results.

    X-axis: layer index. Y-axis: normalized logit difference.
    Two lines: image-only (blue) and text-only (orange).
    Horizontal dashed lines at 0 (no effect) and 1 (full recovery).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = [r["layer"] for r in result["image_only"]]
    image_diffs = [r["normalized_diff"] for r in result["image_only"]]
    text_diffs = [r["normalized_diff"] for r in result["text_only"]]
    image_last16_diffs = [r["normalized_diff"] for r in result["image_last_16"]]
    image_last32_diffs = [r["normalized_diff"] for r in result["image_last_32"]]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(layers, image_diffs, color="blue", label="Image-only patching", linewidth=1.5)
    ax.plot(layers, text_diffs, color="orange", label="Text-only patching", linewidth=1.5)
    ax.plot(layers, image_last16_diffs, color="green", linestyle="--", label="Image last-16 patching", linewidth=1.5)
    ax.plot(layers, image_last32_diffs, color="red", linestyle="--", label="Image last-32 patching", linewidth=1.5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5, label="No effect")
    ax.axhline(y=1, color="gray", linestyle=":", alpha=0.5, label="Full recovery")

    pair = result["pair"]
    ax.set_xlabel("Layer")
    ax.set_ylabel("Normalized logit difference")
    ax.set_title(
        f"Activation Patching: {pair['obj']} "
        f"(count {pair['source_count']} → {pair['target_count']})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Patching plot saved to {output_path}")


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
    parser.add_argument(
        "--activation-patching",
        action="store_true",
        help="Run activation patching experiment on the children pair",
    )
    parser.add_argument(
        "--patch-resize",
        type=int,
        default=448,
        help="Resize dimension for activation patching images (default: 448)",
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

        # Backfill grid dims for old results that lack them
        needs_grid = any(r.get("grid_h") is None for r in results)
        if needs_grid:
            print("Backfilling grid dimensions from dataset images...")
            from datasets import load_dataset

            ds = load_dataset(args.dataset, split="train")
            ps, sms = resolve_patch_params(None, args.model)
            factor = ps * sms
            for r in results:
                if r.get("grid_h") is not None:
                    continue
                idx = r["idx"]
                image = ds[idx]["image"]
                if image is None:
                    continue
                rh, rw = smart_resize(image.height, image.width, factor=factor)
                r["grid_h"] = rh // factor
                r["grid_w"] = rw // factor

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

                # Compute image token grid dimensions
                resized_h, resized_w = smart_resize(
                    image.height, image.width, factor=model.image_factor
                )
                grid_h = resized_h // model.image_factor
                grid_w = resized_w // model.image_factor

                # Step 3: Cache hidden states and forward
                with model.run_with_cache(["lm.blocks.*.hook_resid_post", "lm.blocks.0.hook_resid_pre"]):
                    fwd_output = model.forward(inputs)

                # Step 4: Stack and slice to image tokens only
                activation_cache = ActivationCache()
                activation_cache._data = model.cache
                stacked = activation_cache.stack("lm.blocks.*.hook_resid_post")
                # stacked shape: [num_layers, batch, seq, hidden]
                image_hidden = stacked[:, 0, start_idx : end_idx + 1, :]
                # image_hidden shape: [num_layers, num_image_tokens, hidden]

                # Extract layer-0 input embeddings for register token analysis
                layer0_input = activation_cache["lm.blocks.0.hook_resid_pre"]
                image_embeddings = layer0_input[0, start_idx : end_idx + 1, :]

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

                # Step 8: Register token analysis
                register_info = compute_register_token_analysis(
                    image_embeddings, logit_lens_result
                )

                # Step 9: Ablation analysis
                ablation_info = compute_ablation_analysis(
                    model, inputs, start_idx, end_idx,
                    register_info["number_positions"],
                    norm, lm_head, tokenizer, args.top_k,
                )

                # Step 10: Attention blocking analysis
                original_logits = fwd_output.logits[0, -1, :]
                attn_blocking_info = compute_attention_blocking(
                    model, inputs, start_idx,
                    register_info["number_positions"],
                    original_logits, ground_truth, tokenizer,
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
                    "grid_h": grid_h,
                    "grid_w": grid_w,
                    "number_counts": number_counts_serializable,
                    "rank_unfiltered": rank_unfiltered,
                    "rank_filtered": rank_filtered,
                    "prob_threshold": args.prob_threshold,
                    "register_token_analysis": register_info,
                    "ablation_analysis": ablation_info,
                    "attention_blocking": attn_blocking_info,
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

    # --- Activation patching experiment ---
    if args.activation_patching:
        print("\n" + "=" * 70)
        print("ACTIVATION PATCHING EXPERIMENT")
        print("=" * 70)

        # Ensure model and dataset are available
        if args.analyze_only:
            from datasets import load_dataset

            print(f"Loading dataset for activation patching: {args.dataset}")
            ds = load_dataset(args.dataset, split="train")

            print(f"Loading model for activation patching: {args.model}")
            model = HookedVLM.from_pretrained(args.model)
            components = model.get_model_components()
            tokenizer = components["tokenizer"]

        patching_result = compute_activation_patching(
            model, ds, PATCHING_PAIR, args.patch_resize, tokenizer,
        )

        # Save JSON result
        plots_dir.mkdir(parents=True, exist_ok=True)
        patching_path = output_dir / "activation_patching_children.json"
        with open(patching_path, "w") as f:
            json.dump(patching_result, f, indent=2)
        print(f"\nActivation patching results saved to {patching_path}")

        # Create plot
        create_patching_plot(
            patching_result, plots_dir / "activation_patching_children.png"
        )

        # Print summary
        print(f"\n  Pair: {PATCHING_PAIR['obj']} "
              f"(source count={PATCHING_PAIR['source_count']}, "
              f"target count={PATCHING_PAIR['target_count']})")
        print(f"  Resize: {args.patch_resize}x{args.patch_resize}")
        print(f"  Image tokens: {patching_result['num_image_tokens']}")
        print(f"  Source logit diff: {patching_result['source_logit_diff']}")
        print(f"  Clean target logit diff: {patching_result['clean_logit_diff']}")

        # Summarize where prediction flips occur
        for cond in ["image_only", "text_only", "image_last_16", "image_last_32"]:
            flip_layers = [
                r["layer"] for r in patching_result[cond]
                if r["pred_flipped_to_source"]
            ]
            print(f"  {cond}: prediction flips at {len(flip_layers)} layers"
                  + (f" ({flip_layers})" if flip_layers else ""))

        print("=" * 70)

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

    # Aggregate register token analysis
    register_counts = []
    number_counts_agg = []
    overlap_pcts = []
    for r in results:
        reg = r.get("register_token_analysis")
        if reg is None:
            continue
        reg_pos = set(reg["register_positions"])
        num_pos = set(reg["number_positions"])
        register_counts.append(len(reg_pos))
        number_counts_agg.append(len(num_pos))
        if reg_pos:
            overlap_pcts.append(len(reg_pos & num_pos) / len(reg_pos) * 100)

    register_summary = {}
    if register_counts:
        register_summary = {
            "num_images_with_data": len(register_counts),
            "mean_register_tokens": round(sum(register_counts) / len(register_counts), 2),
            "mean_number_positions": round(sum(number_counts_agg) / len(number_counts_agg), 2),
            "mean_overlap_pct": round(sum(overlap_pcts) / len(overlap_pcts), 2) if overlap_pcts else None,
            "num_images_with_register_tokens": len(overlap_pcts),
        }

    # Aggregate ablation analysis
    ablation_before = []
    ablation_after = []
    for r in results:
        abl = r.get("ablation_analysis")
        if abl is None:
            continue
        ablation_before.append(abl["num_before"])
        ablation_after.append(abl["num_after"])

    ablation_summary = {}
    if ablation_before:
        ablation_summary = {
            "num_images": len(ablation_before),
            "mean_positions_before": round(sum(ablation_before) / len(ablation_before), 2),
            "mean_positions_after": round(sum(ablation_after) / len(ablation_after), 2),
        }

    # Aggregate attention blocking analysis
    attn_block_changed = 0
    attn_block_logprob_changes = []
    attn_block_total = 0
    for r in results:
        ab = r.get("attention_blocking")
        if ab is None:
            continue
        attn_block_total += 1
        if ab["answer_changed"]:
            attn_block_changed += 1
        attn_block_logprob_changes.append(ab["logprob_change"])

    attn_blocking_summary = {}
    if attn_block_total > 0:
        attn_blocking_summary = {
            "num_images": attn_block_total,
            "num_answer_changed": attn_block_changed,
            "mean_logprob_change": round(
                sum(attn_block_logprob_changes) / len(attn_block_logprob_changes), 4
            ),
        }

    # Save aggregate rank results
    plots_dir.mkdir(parents=True, exist_ok=True)
    rank_results = {
        "num_images": len(results),
        "prob_threshold": args.prob_threshold,
        "ranks_unfiltered": ranks_unfiltered,
        "ranks_filtered": ranks_filtered,
        "num_excluded_unfiltered": num_excluded_unfiltered,
        "num_excluded_filtered": num_excluded_filtered,
        "register_token_analysis": register_summary,
        "ablation_analysis": ablation_summary,
        "attention_blocking": attn_blocking_summary,
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

    # Print register token analysis summary
    if register_summary:
        print("\n" + "=" * 70)
        print("REGISTER TOKEN ANALYSIS")
        print("=" * 70)
        print(f"  Images with data:            {register_summary['num_images_with_data']}")
        print(f"  Mean register tokens/image:  {register_summary['mean_register_tokens']}")
        print(f"  Mean number positions/image: {register_summary['mean_number_positions']}")
        if register_summary["mean_overlap_pct"] is not None:
            print(f"  Images with register tokens: {register_summary['num_images_with_register_tokens']}")
            print(f"  Mean overlap %:              {register_summary['mean_overlap_pct']:.1f}%")
            print(f"    (% of register tokens that are also number-token positions)")
        else:
            print("  No images had register tokens — overlap not computable")
        print("=" * 70)

    # Print ablation analysis summary
    if ablation_summary:
        print("\n" + "=" * 70)
        print("ABLATION ANALYSIS (zero ablation at layer 0)")
        print("=" * 70)
        print(f"  Images with data:            {ablation_summary['num_images']}")
        print(f"  Mean positions before:       {ablation_summary['mean_positions_before']}")
        print(f"  Mean positions after:        {ablation_summary['mean_positions_after']}")
        print("=" * 70)

    # Print attention blocking analysis summary
    if attn_blocking_summary:
        print("\n" + "=" * 70)
        print("ATTENTION BLOCKING ANALYSIS (block last→count positions, all layers)")
        print("=" * 70)
        print(f"  Images with data:            {attn_blocking_summary['num_images']}")
        print(f"  Answer changed:              {attn_blocking_summary['num_answer_changed']} / {attn_blocking_summary['num_images']}")
        print(f"  Mean Δ log-prob (correct):   {attn_blocking_summary['mean_logprob_change']}")
        print("=" * 70)

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

    # Create aggregate spatial heatmaps at multiple resolutions
    for res in [128, 64, 32, 16]:
        create_aggregate_heatmap(
            results, plots_dir / f"number_positions_heatmap_{res}x{res}.png", resolution=res
        )


if __name__ == "__main__":
    main()
