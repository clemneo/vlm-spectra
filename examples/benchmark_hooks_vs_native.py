"""Benchmark: HF Native vs Hooks Performance Evaluation.

Measures the timing overhead of the hook-based approach vs HuggingFace native
`output_hidden_states` and `output_attentions` on real CountBench data.

Five benchmark conditions:
  Baseline  - Plain forward, no extras (SDPA/Flash)
  A         - output_hidden_states=True (SDPA/Flash)
  B         - Hooks: hook_resid_pre + hook_resid_post, 64 hooks (SDPA/Flash)
  C         - output_attentions=True (switches to Eager)
  D         - Hooks: attn.hook_pattern, 32 virtual hooks (SDPA/Flash)

Usage:
    uv run python examples/benchmark_hooks_vs_native.py --num-images 5 --warmup 1
    uv run python examples/benchmark_hooks_vs_native.py --num-images 100
    uv run python examples/benchmark_hooks_vs_native.py --output-dir ./results/benchmark
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from tqdm import tqdm

from vlm_spectra import HookedVLM

# ---------------------------------------------------------------------------
# Number-word utilities (for CountBench filtering)
# ---------------------------------------------------------------------------

WORD_TO_NUMBER = {
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9,
}
NUMBER_WORDS = list(WORD_TO_NUMBER.keys())


def parse_count_from_text(text: str) -> int | None:
    """Extract a count from text, trying digits first then number words."""
    m = re.search(r"\b(\d+)\b", text)
    if m:
        return int(m.group(1))
    lower = text.lower()
    for word, val in WORD_TO_NUMBER.items():
        if word in lower:
            return val
    return None


def parse_countbench_item(item: dict) -> tuple[int, str, str | None]:
    """Return (ground_truth_count, caption, object_name_or_None)."""
    caption = item.get("text", item.get("caption", ""))
    number = item.get("number", item.get("count", None))
    if number is not None:
        ground_truth = int(number)
    else:
        parsed = parse_count_from_text(caption)
        ground_truth = parsed if parsed is not None else -1

    object_name = None
    pattern = (
        r"\b(?:of\s+)?(\d+|" + "|".join(NUMBER_WORDS) + r")\s+(.+?)"
        r"(?:\s+on\b|\s+in\b|\s+at\b|\s+with\b|\.|$)"
    )
    match = re.search(pattern, caption.lower())
    if match:
        object_name = match.group(2).strip().rstrip(".")
        if not object_name:
            object_name = None
    return ground_truth, caption, object_name


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def cleanup():
    """Free GPU memory between conditions."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class TimingResult:
    """Aggregated timing statistics for a benchmark condition."""
    condition: str
    description: str
    forward_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)

    @staticmethod
    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
        t = torch.tensor(values)
        return {
            "mean": t.mean().item(),
            "std": t.std().item() if len(values) > 1 else 0.0,
            "median": t.median().item(),
            "min": t.min().item(),
            "max": t.max().item(),
        }

    @property
    def forward_stats(self) -> dict:
        return self._stats(self.forward_ms)

    @property
    def total_stats(self) -> dict:
        return self._stats(self.total_ms)

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "description": self.description,
            "forward_ms": self.forward_ms,
            "total_ms": self.total_ms,
            "forward_stats": self.forward_stats,
            "total_stats": self.total_stats,
        }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_countbench_examples(
    num_images: int, seed: int
) -> list[dict]:
    """Load and filter CountBench examples (count 2-9, non-null images)."""
    from datasets import load_dataset

    ds = load_dataset("nielsr/countbench", split="train")
    total = len(ds)

    valid_indices = []
    for i in range(total):
        item = ds[i]
        if item["image"] is None:
            continue
        gt, _, _ = parse_countbench_item(item)
        if 2 <= gt <= 9:
            valid_indices.append(i)

    print(f"Dataset: {total} total, {len(valid_indices)} valid (count 2-9, non-null)")

    if num_images == -1:
        selected = valid_indices
    else:
        selected = valid_indices[:num_images]

    examples = []
    for idx in selected:
        item = ds[idx]
        gt, caption, obj_name = parse_countbench_item(item)
        if obj_name:
            prompt = f"How many {obj_name} are in this image? Answer with just the number."
        else:
            prompt = "How many objects are in this image? Answer with just the number."
        examples.append({
            "idx": idx,
            "image": item["image"],
            "prompt": prompt,
            "ground_truth": gt,
            "caption": caption,
        })

    print(f"Selected {len(examples)} examples")
    return examples


# ---------------------------------------------------------------------------
# Benchmark condition runners
# ---------------------------------------------------------------------------

def prepare_inputs(model: HookedVLM, example: dict):
    """Prepare model inputs for a single example."""
    return model.prepare_messages(example["prompt"], example["image"])


def run_baseline(model: HookedVLM, inputs) -> float:
    """Plain forward, no extras. Returns forward_ms."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.forward(inputs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def run_condition_a(model: HookedVLM, inputs) -> float:
    """output_hidden_states=True. Returns forward_ms."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.forward(inputs, output_hidden_states=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def run_condition_b(model: HookedVLM, inputs) -> tuple[float, float]:
    """Hooks: hook_resid_pre + hook_resid_post (64 hooks). Returns (forward_ms, total_ms)."""
    hooks = ["lm.blocks.*.hook_resid_pre", "lm.blocks.*.hook_resid_post"]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with model.run_with_cache(hooks):
        torch.cuda.synchronize()
        t_pre = time.perf_counter()
        model.forward(inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    # finalize_cache + format_cache have run here (inside context manager __exit__)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    forward_ms = (t1 - t_pre) * 1000
    total_ms = (t2 - t0) * 1000
    return forward_ms, total_ms


def run_condition_c(model: HookedVLM, inputs) -> float:
    """output_attentions=True (switches to eager). Returns forward_ms."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model.forward(inputs, output_attentions=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000


def run_condition_d(model: HookedVLM, inputs) -> tuple[float, float]:
    """Hooks: attn.hook_pattern (32 virtual hooks, SDPA). Returns (forward_ms, total_ms)."""
    hooks = ["lm.blocks.*.attn.hook_pattern"]

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with model.run_with_cache(hooks):
        torch.cuda.synchronize()
        t_pre = time.perf_counter()
        model.forward(inputs)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
    # finalize_cache (recomputes attention patterns) has run here

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    forward_ms = (t1 - t_pre) * 1000
    total_ms = (t2 - t0) * 1000
    return forward_ms, total_ms


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark_condition(
    name: str,
    description: str,
    run_fn,
    model: HookedVLM,
    examples: list[dict],
    warmup: int,
    returns_total: bool = False,
) -> TimingResult:
    """Run a benchmark condition with warmup, then timed loop over all examples."""
    result = TimingResult(condition=name, description=description)

    # Warmup
    print(f"  Warming up ({warmup} iterations)...")
    for i in range(min(warmup, len(examples))):
        inputs = prepare_inputs(model, examples[i])
        run_fn(model, inputs)
        cleanup()

    # Timed run
    print(f"  Running {len(examples)} examples...")
    for example in tqdm(examples, desc=f"  {name}", leave=False):
        inputs = prepare_inputs(model, example)

        if returns_total:
            forward_ms, total_ms = run_fn(model, inputs)
            result.forward_ms.append(forward_ms)
            result.total_ms.append(total_ms)
        else:
            forward_ms = run_fn(model, inputs)
            result.forward_ms.append(forward_ms)
            result.total_ms.append(forward_ms)

        cleanup()

    stats = result.forward_stats
    print(
        f"  {name}: forward={stats['mean']:.1f} +/- {stats['std']:.1f} ms"
        f"  (median={stats['median']:.1f})"
    )
    if returns_total:
        ts = result.total_stats
        print(
            f"  {name}: total ={ts['mean']:.1f} +/- {ts['std']:.1f} ms"
            f"  (median={ts['median']:.1f})"
        )
    return result


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def print_summary_table(results: dict[str, TimingResult]):
    """Print formatted console summary with overhead calculations."""
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)

    baseline_total = results["Baseline"].total_stats["mean"]

    # Header
    print(
        f"{'Condition':<12} {'Description':<40} "
        f"{'Forward (ms)':<18} {'Total (ms)':<18} {'vs Baseline':<12}"
    )
    print("-" * 100)

    for name, r in results.items():
        fs = r.forward_stats
        ts = r.total_stats
        fwd_str = f"{fs['mean']:>7.1f} +/- {fs['std']:<5.1f}"
        tot_str = f"{ts['mean']:>7.1f} +/- {ts['std']:<5.1f}"
        if baseline_total > 0 and name != "Baseline":
            overhead_pct = (ts["mean"] / baseline_total - 1) * 100
            overhead_str = f"{overhead_pct:>+.1f}%"
        else:
            overhead_str = "---"
        print(
            f"{name:<12} {r.description:<40} "
            f"{fwd_str:<18} {tot_str:<18} {overhead_str:<12}"
        )

    # Wall-clock totals
    print("-" * 100)
    print(
        f"{'Condition':<12} {'':40} {'':18} {'Sum (s)':<18}"
    )
    print("-" * 100)
    for name, r in results.items():
        sum_s = sum(r.total_ms) / 1000
        print(f"{name:<12} {'':40} {'':18} {sum_s:>10.1f} s")

    # Overhead calculations
    print("\n" + "-" * 80)
    print("OVERHEAD ANALYSIS")
    print("-" * 80)

    baseline_fwd = results["Baseline"].forward_stats["mean"]

    if "A" in results and "B" in results:
        a_fwd = results["A"].forward_stats["mean"]
        b_fwd = results["B"].forward_stats["mean"]
        b_tot = results["B"].total_stats["mean"]
        print(f"\nHidden States:")
        print(f"  A (HF native) forward:       {a_fwd:>7.1f} ms  ({a_fwd - baseline_fwd:>+7.1f} ms vs baseline)")
        print(f"  B (hooks) forward:            {b_fwd:>7.1f} ms  ({b_fwd - baseline_fwd:>+7.1f} ms vs baseline)")
        print(f"  B (hooks) total:              {b_tot:>7.1f} ms  ({b_tot - baseline_fwd:>+7.1f} ms vs baseline)")
        if a_fwd > 0:
            print(f"  B total overhead vs A:        {b_tot - a_fwd:>+7.1f} ms  ({(b_tot / a_fwd - 1) * 100:>+.1f}%)")

    if "C" in results and "D" in results:
        c_fwd = results["C"].forward_stats["mean"]
        d_fwd = results["D"].forward_stats["mean"]
        d_tot = results["D"].total_stats["mean"]
        print(f"\nAttention Patterns:")
        print(f"  C (HF eager) forward:         {c_fwd:>7.1f} ms  ({c_fwd - baseline_fwd:>+7.1f} ms vs baseline)")
        print(f"  D (hooks SDPA) forward:       {d_fwd:>7.1f} ms  ({d_fwd - baseline_fwd:>+7.1f} ms vs baseline)")
        print(f"  D (hooks SDPA) total:         {d_tot:>7.1f} ms  ({d_tot - baseline_fwd:>+7.1f} ms vs baseline)")
        if c_fwd > 0:
            print(f"  D total overhead vs C:        {d_tot - c_fwd:>+7.1f} ms  ({(d_tot / c_fwd - 1) * 100:>+.1f}%)")

    if "A" in results and "C" in results:
        a_fwd = results["A"].forward_stats["mean"]
        c_fwd = results["C"].forward_stats["mean"]
        print(f"\nEager vs SDPA penalty:")
        print(f"  C (eager) vs A (SDPA):        {c_fwd - a_fwd:>+7.1f} ms  ({(c_fwd / a_fwd - 1) * 100:>+.1f}%)")

    print()


def save_results_json(results: dict[str, TimingResult], output_dir: Path):
    """Save all per-example timings and statistics as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benchmark_results.json"

    data = {
        "conditions": {name: r.to_dict() for name, r in results.items()},
    }

    # Add summary
    baseline_fwd = results["Baseline"].forward_stats["mean"]
    summary = {"baseline_forward_ms": baseline_fwd}

    if "A" in results and "B" in results:
        summary["hidden_states"] = {
            "A_hf_forward_ms": results["A"].forward_stats["mean"],
            "B_hooks_forward_ms": results["B"].forward_stats["mean"],
            "B_hooks_total_ms": results["B"].total_stats["mean"],
            "B_vs_A_overhead_ms": results["B"].total_stats["mean"] - results["A"].forward_stats["mean"],
        }

    if "C" in results and "D" in results:
        summary["attention_patterns"] = {
            "C_hf_eager_forward_ms": results["C"].forward_stats["mean"],
            "D_hooks_forward_ms": results["D"].forward_stats["mean"],
            "D_hooks_total_ms": results["D"].total_stats["mean"],
            "D_vs_C_overhead_ms": results["D"].total_stats["mean"] - results["C"].forward_stats["mean"],
        }

    data["summary"] = summary

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {output_path}")


def create_comparison_plots(results: dict[str, TimingResult], output_dir: Path):
    """Create box plots comparing conditions (if matplotlib available)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plots")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Hidden states comparison: Baseline, A, B
    ax = axes[0]
    hs_data = []
    hs_labels = []
    for name in ["Baseline", "A", "B"]:
        if name in results:
            vals = results[name].total_ms if name == "B" else results[name].forward_ms
            hs_data.append(vals)
            label = f"{name}\n({results[name].description[:25]})"
            hs_labels.append(label)
    if hs_data:
        bp = ax.boxplot(hs_data, labels=hs_labels, patch_artist=True)
        colors = ["#a8d8ea", "#f6c89f", "#c3e6cb"]
        for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
            patch.set_facecolor(color)
        ax.set_ylabel("Time (ms)")
        ax.set_title("Hidden States: HF Native vs Hooks")
        ax.grid(True, alpha=0.3)

    # Attention comparison: Baseline, C, D
    ax = axes[1]
    attn_data = []
    attn_labels = []
    for name in ["Baseline", "C", "D"]:
        if name in results:
            vals = results[name].total_ms if name == "D" else results[name].forward_ms
            attn_data.append(vals)
            label = f"{name}\n({results[name].description[:25]})"
            attn_labels.append(label)
    if attn_data:
        bp = ax.boxplot(attn_data, labels=attn_labels, patch_artist=True)
        colors = ["#a8d8ea", "#f6c89f", "#c3e6cb"]
        for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
            patch.set_facecolor(color)
        ax.set_ylabel("Time (ms)")
        ax.set_title("Attention Patterns: HF Eager vs Hooks (SDPA)")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "benchmark_comparison.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {plot_path}")


# ---------------------------------------------------------------------------
# Tee helper — write to both stdout and a file
# ---------------------------------------------------------------------------

class TeeStream:
    """Write to both a file and the original stdout."""

    def __init__(self, file: io.TextIOBase, original):
        self._file = file
        self._original = original

    def write(self, data):
        self._original.write(data)
        self._file.write(data)

    def flush(self):
        self._original.flush()
        self._file.flush()


# ---------------------------------------------------------------------------
# Run all five conditions on a given slice of examples
# ---------------------------------------------------------------------------

def run_all_conditions(
    model: HookedVLM,
    examples: list[dict],
    warmup: int,
    output_dir: Path,
    n_label: int,
) -> dict[str, TimingResult]:
    """Run the five benchmark conditions and save results for *n_label* images."""
    num_layers = model.lm_num_layers
    results: dict[str, TimingResult] = {}

    print(f"\n--- Baseline: Plain forward ---")
    results["Baseline"] = run_benchmark_condition(
        "Baseline", "Plain forward, no extras (SDPA)", run_baseline,
        model, examples, warmup,
    )
    cleanup()

    print(f"\n--- Condition A: output_hidden_states=True ---")
    results["A"] = run_benchmark_condition(
        "A", "output_hidden_states=True (SDPA)", run_condition_a,
        model, examples, warmup,
    )
    cleanup()

    print(f"\n--- Condition B: Hooks resid_pre + resid_post ({num_layers * 2} hooks) ---")
    results["B"] = run_benchmark_condition(
        "B", f"Hooks: resid_pre+post ({num_layers * 2} hooks, SDPA)", run_condition_b,
        model, examples, warmup, returns_total=True,
    )
    cleanup()

    print(f"\n--- Condition C: output_attentions=True (eager) ---")
    results["C"] = run_benchmark_condition(
        "C", "output_attentions=True (Eager)", run_condition_c,
        model, examples, warmup,
    )
    cleanup()

    print(f"\n--- Condition D: Hooks attn.hook_pattern ({num_layers} virtual hooks) ---")
    results["D"] = run_benchmark_condition(
        "D", f"Hooks: attn.hook_pattern ({num_layers} virtual, SDPA)", run_condition_d,
        model, examples, warmup, returns_total=True,
    )
    cleanup()

    print_summary_table(results)

    sub_dir = output_dir / f"n{n_label}"
    save_results_json(results, sub_dir)
    create_comparison_plots(results, sub_dir)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark HF native vs hooks performance on CountBench"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llava-hf/llava-1.5-7b-hf",
        help="Model name (default: llava-hf/llava-1.5-7b-hf)",
    )
    parser.add_argument(
        "--num-images",
        type=str,
        default="100,200,300",
        help="Comma-separated list of image counts to benchmark (default: 100,200,300)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations per condition",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/benchmark",
        help="Directory for output files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    # Parse image counts
    image_counts = [int(x.strip()) for x in args.num_images.split(",")]
    max_images = max(image_counts)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tee stdout to a log file
    log_path = output_dir / "benchmark_output.txt"
    log_file = open(log_path, "w")
    original_stdout = sys.stdout
    sys.stdout = TeeStream(log_file, original_stdout)

    try:
        torch.manual_seed(args.seed)

        # Load the maximum number of examples needed
        print("Loading CountBench examples...")
        all_examples = load_countbench_examples(max_images, args.seed)

        # Load model
        print(f"\nLoading model: {args.model}")
        model = HookedVLM.from_pretrained(args.model)
        print(f"Model loaded: {model.lm_num_layers} layers, device={model.device}")
        print(f"Attention implementation: {model.adapter._original_attn_impl}")

        # Run benchmark for each image count
        for n in image_counts:
            examples = all_examples[:n]
            actual_n = len(examples)

            print("\n" + "#" * 80)
            print(f"# BENCHMARK RUN: {actual_n} images")
            print("#" * 80)
            print(f"\nBenchmark configuration:")
            print(f"  Model: {args.model}")
            print(f"  Layers: {model.lm_num_layers}")
            print(f"  Images: {actual_n} (requested {n})")
            print(f"  Warmup: {args.warmup}")

            run_all_conditions(model, examples, args.warmup, output_dir, actual_n)

        print(f"\nAll output saved to {log_path}")
    finally:
        sys.stdout = original_stdout
        log_file.close()

    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
