"""Utility to pre-download every VLM checkpoint used by the suite.

Run with ``uv run tests/download_models.py`` to populate the Hugging Face
cache before executing a heavy pytest target. Hugging Face will display its
native download progress bars for each file.
"""

from __future__ import annotations

import argparse
import gc
import sys
from typing import Any, Dict

import torch

from vlm_spectra.models.registry import ModelRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and cache all VLM checkpoints upfront."
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Specific model name to download (can be passed multiple times).",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Force downloading every supported model regardless of --model.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the supported model names and exit.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Skip models that fail to load instead of aborting immediately.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map forwarded to the underlying Hugging Face loader.",
    )
    parser.add_argument(
        "--torch-dtype",
        default="bfloat16",
        help="Torch dtype string for the HF loader (default: bfloat16).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    supported = ModelRegistry.list_supported_models()
    try:
        torch_dtype = getattr(torch, args.torch_dtype)
    except AttributeError:
        print(f"Unknown torch dtype: {args.torch_dtype}")
        return 2

    if args.list_models:
        print("Supported models:")
        for name in supported:
            print(f" - {name}")
        return 0

    if args.all_models or not args.model:
        targets = supported
    else:
        targets = args.model

    if not targets:
        print("No models registered; nothing to download.")
        return 0

    failed: list[str] = []
    total = len(targets)
    for idx, name in enumerate(targets, start=1):
        print(f"[{idx}/{total}] Preloading {name}", flush=True)
        try:
            adapter_cls = ModelRegistry.get_adapter_class(name)
            hf_kwargs: Dict[str, Any] = {"torch_dtype": torch_dtype}
            if args.device is not None:
                hf_kwargs["device_map"] = args.device

            model = adapter_cls.MODEL_CLASS.from_pretrained(
                name,
                **hf_kwargs,
            )
            processor = adapter_cls.PROCESSOR_CLASS.from_pretrained(name)
        except Exception as exc:  # pragma: no cover - runtime safety net
            print(f"    Failed to load {name}: {exc}")
            failed.append(name)
            if not args.continue_on_error:
                return 1
            continue

        del model, processor
        gc.collect()
        if torch.cuda.is_available():  # keep GPU memory usage bounded
            torch.cuda.empty_cache()

    if failed:
        print(f"Completed with failures: {', '.join(failed)}")
        return 1

    print("All requested models cached.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
