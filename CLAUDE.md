# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VLM Spectra is a research toolkit for Vision-Language Model (VLM) interpretability, with a focus on logit lens analysis. The project provides tools to analyze how VLMs process images at different layers, particularly for GUI automation tasks using the UI-TARS model.

## Development Commands

### Package Management
- **Install dependencies**: `uv sync` (uses uv.lock for reproducible builds)
- **Install in development mode**: `uv pip install -e .`

### Testing
- **Run all tests**: `uv run pytest`
- **Run specific test file**: `uv run pytest tests/test_filename.py`
- **Run with coverage**: `uv run pytest --cov=src/vlm_spectra`

### Code Quality
- **Lint code**: `uv run ruff check src/ tests/`
- **Format code**: `uv run ruff format src/ tests/`

## Architecture

### Core Components

**HookedVLM** (`src/vlm_spectra/models/HookedVLM.py`): The main model wrapper that:
- Wraps Hugging Face VLMs with interpretability hooks
- Currently supports ByteDance-Seed/UI-TARS-1.5-7B (Qwen2.5-VL based)
- Provides `generate()` and `forward()` methods with optional hidden state extraction
- Includes `get_model_components()` for accessing internal layers

**Logit Lens** (`src/vlm_spectra/logit_lens/create_logit_lens.py`):
- Generates interactive HTML visualizations showing token predictions at each layer
- Maps image patches to token predictions for visual analysis
- Creates standalone HTML files with embedded JavaScript for portability

**Utils** (`src/vlm_spectra/utils/qwen_25_vl_utils.py`):
- Image/video preprocessing functions specific to Qwen2.5-VL
- Smart resizing algorithms that maintain aspect ratios
- Vision input processing utilities

### Model Integration Pattern

The project uses a metadata-driven approach for supporting multiple VLM architectures:
- `vlm_metadata.py` contains architecture-specific information
- Extensible design allows adding new models by updating metadata
- Current focus on Qwen2.5-VL but structured for multi-model support

### GUI Automation Specialization

The UI-TARS model is specialized for GUI automation with:
- Specific prompt templates for UI interaction tasks
- Action space including click, drag, type, scroll operations
- Screenshot analysis capabilities for UI element detection

## Testing Strategy

Tests focus on core functionality verification:
- Model initialization and component access
- Generation with/without hidden state extraction
- Forward pass functionality
- Logit lens visualization creation

## Development Notes

- The project uses src-layout packaging structure
- Python 3.10+ required for modern typing features
- PyTorch ecosystem with Transformers for model handling
- Research-oriented codebase optimized for interpretability rather than production deployment