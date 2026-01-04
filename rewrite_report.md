# Rewrite Report

## Notes
- `tests/test_batching.py::TestBatchMethods::test_batch_gradients[ByteDance-Seed/UI-TARS-1.5-7B-True]` hits CUDA OOM during `loss.backward()` on this lower-memory node. This is expected per local GPU limits.

## Test Runs
- `uv run pytest tests/hookedvlm_test.py -v` (pass)
- `uv run pytest tests/test_run_with_cache.py -v` (pass)
- `uv run pytest tests/test_prefill_functionality.py -v` (pass)
- `uv run pytest tests/test_batching.py -v` (all pass except the CUDA OOM noted above)
- `uv run pytest -v` (timed out after 180s; up to 83% complete, only observed failure was the same CUDA OOM test)
