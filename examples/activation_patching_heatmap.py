"""
Activation Patching Example: Color Discrimination in VLMs

This example demonstrates activation patching to identify which (layer, position)
combinations encode color information. We create two images (red vs blue patch)
and measure how patching activations affects the model's prediction.

Usage:
    # Debug mode - check model predictions first
    uv run python examples/activation_patching_heatmap.py --debug

    # Full patching experiment
    uv run python examples/activation_patching_heatmap.py

    # Patch only image tokens (faster)
    uv run python examples/activation_patching_heatmap.py --image-tokens-only
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from vlm_spectra import HookedVLM
from vlm_spectra.hooks.patch_hooks import PatchResidualHook
from vlm_spectra.preprocessing.utils.vision_info import process_vision_info


def create_colored_patch_image(
    size: int = 224,
    patch_size: int = 28,
    center_color: Tuple[int, int, int] = (0, 255, 0),
    background_color: Tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """
    Create an image with a colored center patch.

    Args:
        size: Total image size (width and height)
        patch_size: Size of each patch (image will be size/patch_size patches)
        center_color: RGB tuple for the center patch
        background_color: RGB tuple for all other patches

    Returns:
        PIL Image with the colored center patch
    """
    image = Image.new("RGB", (size, size), background_color)
    draw = ImageDraw.Draw(image)

    # Calculate center patch position (for an 8x8 grid, center is patch index 36)
    num_patches = size // patch_size  # 8 patches per dimension
    center_row = num_patches // 2  # Row 4 (middle)
    center_col = num_patches // 2  # Col 4 (middle)

    # Draw center patch
    x1 = center_col * patch_size
    y1 = center_row * patch_size
    x2 = x1 + patch_size
    y2 = y1 + patch_size
    draw.rectangle([x1, y1, x2, y2], fill=center_color)

    return image


def create_experiment_images() -> Tuple[Image.Image, Image.Image]:
    """Create the clean (red) and corrupted (blue) images."""
    red_image = create_colored_patch_image(
        size=224,
        patch_size=28,
        center_color=(255, 0, 0),  # Red
        background_color=(128, 128, 128),  # Gray
    )

    blue_image = create_colored_patch_image(
        size=224,
        patch_size=28,
        center_color=(0, 0, 255),  # Blue
        background_color=(128, 128, 128),  # Gray
    )

    return red_image, blue_image


def get_activations_for_all_layers(
    model: HookedVLM,
    inputs: Dict[str, torch.Tensor],
) -> Dict[Tuple[str, int], torch.Tensor]:
    """
    Capture residual stream activations at all layers.

    Args:
        model: HookedVLM instance
        inputs: Prepared model inputs

    Returns:
        Dictionary mapping (hook_name, layer) to activation tensors
    """
    with model.run_with_cache(["lm.layer.post"]):
        model.forward(inputs)

    # Return a copy of the cache to preserve activations
    return {key: tensor.clone() for key, tensor in model.cache.items()}


def get_token_ids_for_colors(model: HookedVLM) -> Tuple[int, int]:
    """
    Get token IDs for ' red' and ' blue' tokens (leading space).

    Returns:
        Tuple of (red_token_id, blue_token_id)
    """
    components = model.get_model_components()
    tokenizer = components["tokenizer"]

    # Get token IDs - use a leading space for word-boundary tokens
    red_id = tokenizer.encode(" red", add_special_tokens=False)[0]
    blue_id = tokenizer.encode(" blue", add_special_tokens=False)[0]

    return red_id, blue_id


def compute_logit_difference(
    model: HookedVLM,
    inputs: Dict[str, torch.Tensor],
    red_token_id: int,
    blue_token_id: int,
) -> float:
    """
    Compute logit(red) - logit(blue) for the next token prediction.

    Args:
        model: HookedVLM instance
        inputs: Prepared model inputs
        red_token_id: Token ID for "red"
        blue_token_id: Token ID for "blue"

    Returns:
        Logit difference (red - blue)
    """
    logit_diff, _ = compute_metrics(model, inputs, red_token_id, blue_token_id)
    return logit_diff


def compute_blue_probability(
    model: HookedVLM,
    inputs: Dict[str, torch.Tensor],
    red_token_id: int,
    blue_token_id: int,
) -> float:
    """
    Compute the probability for the blue token in the next token prediction.

    Args:
        model: HookedVLM instance
        inputs: Prepared model inputs
        red_token_id: Token ID for "red"
        blue_token_id: Token ID for "blue"

    Returns:
        Probability of the blue token
    """
    _, blue_prob = compute_metrics(model, inputs, red_token_id, blue_token_id)
    return blue_prob


def compute_metrics(
    model: HookedVLM,
    inputs: Dict[str, torch.Tensor],
    red_token_id: int,
    blue_token_id: int,
) -> Tuple[float, float]:
    """
    Compute logit(red) - logit(blue) and blue probability for the next token.

    Returns:
        Tuple of (logit difference, blue probability)
    """
    outputs = model.forward(inputs)
    logits = outputs.logits[0, -1, :]  # Last token position
    probs = torch.softmax(logits, dim=-1)

    red_logit = logits[red_token_id].item()
    blue_logit = logits[blue_token_id].item()
    blue_prob = probs[blue_token_id].item()

    return red_logit - blue_logit, blue_prob


def debug_mode(
    model: HookedVLM,
    red_inputs: Dict[str, torch.Tensor],
    blue_inputs: Dict[str, torch.Tensor],
    red_token_id: int,
    blue_token_id: int,
) -> None:
    """
    Debug mode: print top token probabilities for each image.

    Args:
        model: HookedVLM instance
        red_inputs: Inputs for the red image
        blue_inputs: Inputs for the blue image
        red_token_id: Token ID for "red"
        blue_token_id: Token ID for "blue"
    """
    components = model.get_model_components()
    tokenizer = components["tokenizer"]

    print("=" * 60)
    print("DEBUG MODE: Checking model predictions")
    print("=" * 60)

    for name, inputs in [("RED image", red_inputs), ("BLUE image", blue_inputs)]:
        outputs = model.forward(inputs)
        logits = outputs.logits[0, -1, :]  # Last token position

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Top 10 predictions
        top_probs, top_ids = probs.topk(10)

        print(f"\n{name} - Top 10 predictions:")
        print("-" * 40)
        for prob, tok_id in zip(top_probs, top_ids):
            token_str = tokenizer.decode(tok_id.item())
            print(f"  {token_str:15s} {prob.item():.4f}")

        # Specifically show red and blue
        red_prob = probs[red_token_id].item()
        blue_prob = probs[blue_token_id].item()
        red_logit = logits[red_token_id].item()
        blue_logit = logits[blue_token_id].item()

        print(f"\n  Specific tokens:")
        print(f"    ' red': prob={red_prob:.4f}, logit={red_logit:.2f}")
        print(f"    ' blue': prob={blue_prob:.4f}, logit={blue_logit:.2f}")
        print(f"    logit diff (red-blue): {red_logit - blue_logit:.2f}")

    print("\n" + "=" * 60)


def run_activation_patching(
    model: HookedVLM,
    clean_inputs: Dict[str, torch.Tensor],
    corrupted_cache: Dict[Tuple[str, int], torch.Tensor],
    red_token_id: int,
    blue_token_id: int,
    token_positions: List[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, List[int]]:
    """
    Run activation patching experiment.

    For each (layer, position), patch the corrupted (blue) activation into the
    clean (red) forward pass and measure the logit difference change.

    Args:
        model: HookedVLM instance
    clean_inputs: Inputs for the clean (red) image
    corrupted_cache: Cached activations from corrupted (blue) image
    red_token_id: Token ID for "red"
    blue_token_id: Token ID for "blue"
        token_positions: Optional list of token positions to patch (None = all)

    Returns:
        Tuple of (logit-diff results, blue-prob results, baseline logit diff,
        baseline blue prob, token positions used)
    """
    num_layers = model.lm_num_layers

    # Determine sequence length from cached activations
    sample_activation = corrupted_cache[("lm.layer.post", 0)]
    seq_len = sample_activation.shape[1]

    # Default to all positions if not specified
    if token_positions is None:
        token_positions = list(range(seq_len))

    num_positions = len(token_positions)

    # Get baseline logit difference (clean forward pass, no patching)
    baseline_logit_diff, baseline_blue_prob = compute_metrics(
        model, clean_inputs, red_token_id, blue_token_id
    )
    print(f"Baseline logit difference (red - blue): {baseline_logit_diff:.4f}")
    print(f"Baseline blue probability: {baseline_blue_prob:.4f}")

    # Initialize results matrix
    results = np.zeros((num_layers, num_positions))
    blue_prob_results = np.zeros((num_layers, num_positions))

    # Patching loop
    total_iterations = num_layers * num_positions
    with tqdm(total=total_iterations, desc="Patching activations") as pbar:
        for layer_idx in range(num_layers):
            for pos_idx, token_pos in enumerate(token_positions):
                # Get the corrupted activation at this (layer, position)
                corrupted_activation = corrupted_cache[("lm.layer.post", layer_idx)]
                replacement = corrupted_activation[0, token_pos, :].clone()

                # Create patch hook
                hook = PatchResidualHook(
                    layer=layer_idx,
                    token_idx=token_pos,
                    replacement=replacement,
                )

                # Run forward pass with patching
                with model.run_with_hooks([hook]):
                    patched_logit_diff, patched_blue_prob = compute_metrics(
                        model, clean_inputs, red_token_id, blue_token_id
                    )

                # Store the change from baseline
                # Negative value means patching pushed toward "blue" (corrupted)
                results[layer_idx, pos_idx] = patched_logit_diff - baseline_logit_diff
                blue_prob_results[layer_idx, pos_idx] = (
                    patched_blue_prob - baseline_blue_prob
                )

                pbar.update(1)

    return (
        results,
        blue_prob_results,
        baseline_logit_diff,
        baseline_blue_prob,
        token_positions,
    )


def create_heatmap_visualization(
    results: np.ndarray,
    token_positions: List[int],
    baseline_value: float,
    model_name: str,
    output_path: str = "activation_patching_heatmap.png",
    token_labels: List[str] | None = None,
    colorbar_label: str = "Logit Difference Change (patched - baseline)",
    cmap: str = "RdBu_r",
    title_lines: List[str] | None = None,
) -> None:
    """
    Create and save a heatmap visualization of activation patching results.

    Args:
        results: 2D array of shape (num_layers, num_positions)
        token_positions: List of token position indices
        baseline_logit_diff: Baseline logit difference for reference
        model_name: Name of the model (for title)
        output_path: Path to save the visualization
        token_labels: Optional labels for token positions
    """
    num_layers, num_positions = results.shape

    fig, ax = plt.subplots(
        figsize=(max(12, num_positions * 0.5), max(8, num_layers * 0.3))
    )

    # Create heatmap with diverging colormap
    vmax = max(abs(results.min()), abs(results.max()))
    vmin = -vmax

    im = ax.imshow(
        results,
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label, fontsize=12)

    # Set axis labels
    ax.set_xlabel("Token Position", fontsize=12)
    ax.set_ylabel("Layer", fontsize=12)

    # Set x-ticks
    if token_labels is not None:
        ax.set_xticks(range(num_positions))
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    else:
        # Show subset of ticks if many positions
        if num_positions > 20:
            tick_step = num_positions // 10
            ax.set_xticks(range(0, num_positions, tick_step))
            ax.set_xticklabels(
                [str(token_positions[i]) for i in range(0, num_positions, tick_step)]
            )
        else:
            ax.set_xticks(range(num_positions))
            ax.set_xticklabels([str(p) for p in token_positions])

    # Set y-ticks
    if num_layers > 20:
        tick_step = num_layers // 10
        ax.set_yticks(range(0, num_layers, tick_step))
    else:
        ax.set_yticks(range(num_layers))

    # Title
    if title_lines is None:
        title_lines = [
            f"Activation Patching: {model_name}",
            f"Baseline value: {baseline_value:.2f}",
        ]
    ax.set_title("\n".join(title_lines), fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Heatmap saved to: {output_path}")


def create_summary_visualization(
    results: np.ndarray,
    token_positions: List[int],
    image_token_range: Tuple[int, int],
    output_path: str = "activation_patching_summary.png",
) -> None:
    """
    Create a summary visualization showing aggregate effects.

    Args:
        results: 2D array of shape (num_layers, num_positions)
        token_positions: List of token position indices
        image_token_range: (start, end) indices of image tokens
        output_path: Path to save the visualization
    """
    num_layers = results.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Average effect per layer (averaged over positions)
    layer_effects = np.mean(results, axis=1)
    axes[0].barh(range(num_layers), layer_effects, color="steelblue")
    axes[0].set_xlabel("Average Logit Diff Change")
    axes[0].set_ylabel("Layer")
    axes[0].set_title("Effect by Layer\n(averaged over positions)")
    axes[0].axvline(x=0, color="black", linestyle="--", linewidth=0.5)

    # Plot 2: Average effect per position type
    start_img, end_img = image_token_range

    # Categorize positions
    position_categories = []
    category_effects = []

    # Image tokens
    img_mask = [(p >= start_img and p <= end_img) for p in token_positions]
    if any(img_mask):
        img_indices = [i for i, m in enumerate(img_mask) if m]
        img_effect = np.mean(results[:, img_indices])
        position_categories.append("Image\nTokens")
        category_effects.append(img_effect)

    # Text tokens (before image)
    pre_img_mask = [(p < start_img) for p in token_positions]
    if any(pre_img_mask):
        pre_indices = [i for i, m in enumerate(pre_img_mask) if m]
        pre_effect = np.mean(results[:, pre_indices])
        position_categories.append("Pre-Image\nText")
        category_effects.append(pre_effect)

    # Text tokens (after image)
    post_img_mask = [(p > end_img) for p in token_positions]
    if any(post_img_mask):
        post_indices = [i for i, m in enumerate(post_img_mask) if m]
        post_effect = np.mean(results[:, post_indices])
        position_categories.append("Post-Image\nText")
        category_effects.append(post_effect)

    colors = ["red" if e < 0 else "gray" for e in category_effects]
    axes[1].bar(position_categories, category_effects, color=colors)
    axes[1].set_ylabel("Average Logit Diff Change")
    axes[1].set_title("Effect by Token Type\n(averaged over layers)")
    axes[1].axhline(y=0, color="black", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Summary saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Activation Patching Example: Identify where color information is encoded"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Model name to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tmp/activation_patching_results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--image-tokens-only",
        action="store_true",
        help="Only patch image token positions (faster)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: print top predictions for each image and exit",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model: {args.model}")
    model = HookedVLM.from_pretrained(args.model)
    print(f"Model has {model.lm_num_layers} layers")

    # Create experiment images
    print("Creating experiment images...")
    red_image, blue_image = create_experiment_images()

    # Save images for reference
    red_image.save(output_dir / "red_image.png")
    blue_image.save(output_dir / "blue_image.png")
    print(f"Saved experiment images to {output_dir}")

    # Prepare inputs without the computer-use system prompt
    prompt = "Describe this image"
    assistant_prefill = "There is a small"

    def prepare_inputs_without_tools(image: Image.Image) -> Dict[str, torch.Tensor]:
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]

        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})

        if assistant_prefill:
            rendered_text = model.processor.apply_chat_template(
                messages, tokenize=False, continue_final_message=True
            )
        else:
            rendered_text = model.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        image_inputs, video_inputs = process_vision_info(messages)
        return model.processor(
            text=[rendered_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    red_inputs = prepare_inputs_without_tools(red_image)
    blue_inputs = prepare_inputs_without_tools(blue_image)

    # Get token IDs for red and blue
    red_token_id, blue_token_id = get_token_ids_for_colors(model)
    print(f"Token IDs - red: {red_token_id}, blue: {blue_token_id}")

    # Debug mode: print predictions and exit
    if args.debug:
        debug_mode(model, red_inputs, blue_inputs, red_token_id, blue_token_id)
        return

    # Get image token range
    image_token_range = model.get_image_token_range(red_inputs)
    print(f"Image token range: {image_token_range}")

    # Capture corrupted (blue) activations
    print("Capturing corrupted (blue) activations...")
    corrupted_cache = get_activations_for_all_layers(model, blue_inputs)

    # Determine which positions to patch
    seq_len = corrupted_cache[("lm.layer.post", 0)].shape[1]
    if args.image_tokens_only:
        start, end = image_token_range
        token_positions = list(range(start, end + 1))
        print(f"Patching only image tokens: positions {start} to {end}")
    else:
        token_positions = list(range(seq_len))
        print(f"Patching all {seq_len} token positions")
    positions = token_positions

    # Generate token labels for visualization
    components = model.get_model_components()
    tokenizer = components["tokenizer"]
    input_ids = red_inputs["input_ids"][0]
    token_labels = []
    img_start, img_end = image_token_range
    img_counter = 0
    for i, token_id in enumerate(input_ids):
        if i >= img_start and i <= img_end:
            img_counter += 1
            token_labels.append(f"<IMG{img_counter:03d}>")
        else:
            decoded = tokenizer.decode(token_id.item())
            # Truncate long tokens for display
            if len(decoded) > 8:
                decoded = decoded[:6] + ".."
            token_labels.append(decoded)

    # Run activation patching
    print("Running activation patching experiment...")
    (
        results,
        blue_prob_results,
        _baseline_logit_diff,
        _baseline_blue_prob,
        _positions,
    ) = run_activation_patching(
        model=model,
        clean_inputs=red_inputs,
        corrupted_cache=corrupted_cache,
        red_token_id=red_token_id,
        blue_token_id=blue_token_id,
        token_positions=token_positions,
    )

    # Save raw results
    np.save(output_dir / "patching_results.npy", results)
    np.save(output_dir / "patching_results_blue_prob.npy", blue_prob_results)

    # Compute baselines for titles and summaries
    baseline_logit_diff = compute_logit_difference(
        model, red_inputs, red_token_id, blue_token_id
    )
    baseline_blue_prob = compute_blue_probability(
        model, red_inputs, red_token_id, blue_token_id
    )

    # Create visualizations
    print("Creating visualizations...")

    # Filter token labels to match positions
    position_labels = [
        token_labels[p] if p < len(token_labels) else f"pos_{p}" for p in positions
    ]

    create_heatmap_visualization(
        results=results,
        token_positions=positions,
        baseline_value=baseline_logit_diff,
        model_name=args.model.split("/")[-1],
        output_path=str(output_dir / "activation_patching_heatmap.png"),
        token_labels=position_labels,
        colorbar_label="Logit Difference Change (patched - baseline)",
        cmap="RdBu_r",
        title_lines=[
            f"Activation Patching: {args.model.split('/')[-1]}",
            f"Baseline logit diff (red-blue): {baseline_logit_diff:.2f}",
            "Blue regions = patching changed prediction toward 'blue'",
        ],
    )

    create_heatmap_visualization(
        results=blue_prob_results,
        token_positions=positions,
        baseline_value=baseline_blue_prob,
        model_name=args.model.split("/")[-1],
        output_path=str(output_dir / "activation_patching_blue_prob_heatmap.png"),
        token_labels=position_labels,
        colorbar_label="Blue Probability Change (patched - baseline)",
        cmap="RdBu_r",
        title_lines=[
            f"Activation Patching: {args.model.split('/')[-1]}",
            f"Baseline blue probability: {baseline_blue_prob:.4f}",
            "Blue regions = patching increased blue probability",
        ],
    )
    create_summary_visualization(
        results=results,
        token_positions=positions,
        image_token_range=image_token_range,
        output_path=str(output_dir / "activation_patching_summary.png"),
    )

    # Print summary statistics
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Baseline logit difference (red - blue): {baseline_logit_diff:.4f}")
    print(f"Most influential position: {positions[np.argmin(results.min(axis=0))]}")
    print(f"Most influential layer: {np.argmin(results.min(axis=1))}")

    # Find top-5 most influential (layer, position) pairs
    flat_results = results.flatten()
    top_indices = np.argsort(flat_results)[:5]  # Most negative = most influential
    print("\nTop 5 most influential (layer, position) pairs:")
    for idx in top_indices:
        layer = idx // len(positions)
        pos_idx = idx % len(positions)
        pos = positions[pos_idx]
        effect = flat_results[idx]
        print(f"  Layer {layer}, Position {pos} ({position_labels[pos_idx]}): {effect:.4f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
