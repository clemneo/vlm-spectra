# VLM Spectra

A research toolkit for Vision-Language Model (VLM) interpretability, with a focus on logit lens analysis and model introspection.

## HookedVLM: Interpretable VLM Wrapper

The core of VLM Spectra is the `HookedVLM` class, which wraps Hugging Face Vision-Language Models with interpretability hooks. This allows researchers to extract hidden states, attention patterns, and perform detailed analysis of how VLMs process visual and textual information.

### Supported Models

Currently supports:
- **ByteDance-Seed/UI-TARS-1.5-7B** (Qwen2.5-VL based) - Specialized for GUI automation tasks

The architecture is designed to be extensible for additional VLM models.

### Installation

```bash
# One-line install
uv pip install git+https://github.com/clemneo/vlm-spectra.git

# Or for development
git clone https://github.com/clemneo/vlm-spectra.git
cd vlm-spectra
uv sync
```

### Pre-download Model Weights

The acceptance tests can target several large checkpoints. Run the preload helper
once to fetch every supported model into your local Hugging Face cache (progress
bars are provided by Hugging Face):

```bash
uv run tests/download_models.py
```

Use `--model` to pull a specific checkpoint, `--all-models` to force refreshing
everything, or `--list-models` to inspect what’s available. The helper talks
directly to the underlying Hugging Face models/processors, so it still works if
`HookedVLM` happens to be broken during development.

### Basic Usage

```python
from vlm_spectra import HookedVLM
from PIL import Image

# Initialize the model
model = HookedVLM.from_pretrained("ByteDance-Seed/UI-TARS-1.5-7B")

# Load an image
image = Image.open("screenshot.png")

# Prepare inputs for the model
inputs = model.prepare_messages("Click on the login button", image)

# Generate response
outputs = model.generate(inputs, max_new_tokens=100)
print(model.processor.decode(outputs.sequences[0]))
```

### Key Methods

#### `prepare_messages(task, image, assistant_prefill="", return_text=False)`
Prepares inputs for the model by formatting the task and image according to the model's expected format.

```python
# Basic usage
inputs = model.prepare_messages("Describe what you see", image)

# With assistant prefill (guides the response)
inputs = model.prepare_messages(
    "Click on the red button",
    image,
    assistant_prefill='Thought: I see that there is a'
)

# Return both inputs and formatted text
inputs, text = model.prepare_messages("Click here", image, return_text=True)
```

#### `generate(inputs, max_new_tokens=512, output_hidden_states=False, **kwargs)`
Generates text responses from the model with optional hidden state extraction.

```python
# Standard generation
outputs = model.generate(inputs, max_new_tokens=100)

# Generate with hidden states for analysis
outputs = model.generate(inputs, output_hidden_states=True)
hidden_states = outputs.hidden_states  # Tuple of tensors for each layer
```

#### `forward(inputs, output_hidden_states=False, output_attentions=False, **kwargs)`
Runs a forward pass through the model without generation, useful for analysis.

```python
# Forward pass with hidden states
outputs = model.forward(inputs, output_hidden_states=True)

# Forward pass with attention patterns
outputs = model.forward(inputs, output_attentions=True)
attention_weights = outputs.attentions
```

#### `run_with_cache(hook_names)`
Context manager for caching specific model components during forward/generate calls.

```python
# Cache attention patterns and residual outputs
with model.run_with_cache(["lm.blocks.*.attn.hook_pattern", "lm.blocks.*.hook_resid_post"]) as cache:
    outputs = model.forward(inputs)

# Access cached values
attention_cache = cache["lm.blocks.5.attn.hook_pattern"]
residual_cache = cache["lm.blocks.5.hook_resid_post"]
# Or stack all layers
all_residuals = cache.stack("lm.blocks.*.hook_resid_post")
```

### Advanced Analysis

#### Logit Lens Visualization
Create interactive HTML visualizations showing how token predictions evolve across layers:

```python
from vlm_spectra.logit_lens.create_logit_lens import create_logit_lens
from vlm_spectra.models.vlm_metadata import VLMMetadataExtractor

# Run forward pass with hidden states
outputs = model.forward(inputs, output_hidden_states=True)

# Extract metadata for visualization
metadata = VLMMetadataExtractor.extract_metadata_qwen(
    model.model, model.processor, inputs, image
)

# Create interactive visualization
create_logit_lens(
    hidden_states=outputs.hidden_states,
    norm=model.model.model.norm,
    lm_head=model.model.lm_head,
    tokenizer=model.processor.tokenizer,
    image=image,
    token_labels=metadata['token_labels'],
    image_size=metadata['image_size'],
    grid_size=metadata['grid_size'],
    patch_size=metadata['patch_size'],
    model_name="UI-TARS",
    image_filename="screenshot.png",
    prompt="Click on the login button"
)
```

#### Batch Processing
Process multiple images efficiently:

```python
# Prepare batch inputs
tasks = ["Click button", "Describe image", "Find text field"]
images = [img1, img2, img3]

batch_inputs = model.prepare_messages_batch(tasks, images)
batch_outputs = model.generate_batch(batch_inputs)
```

## Web Application Interface

VLM Spectra includes a Flask-based web application that provides an interactive interface for model analysis without requiring code. This is particularly useful for researchers who want to quickly test the model or demonstrate its capabilities.

### Starting the Web App

```bash
cd src/vlm_spectra/web_app
python app.py
```

The application will be available at `http://localhost:55556`.

### Features

#### Image Upload and Analysis
- **Drag-and-drop interface** for uploading screenshots or images
- **Task specification** with natural language descriptions
- **Assistant prefill** to guide model responses toward specific formats (e.g., JSON)
- **Real-time prediction** with coordinate extraction for GUI elements

#### Interactive Analysis Tools

**1. Forward Pass Analysis**
- Examine model outputs without full generation
- View top predicted tokens at each position
- Analyze confidence scores and probability distributions

**2. Attention Visualization**
- Layer-by-layer attention pattern analysis
- Head-specific attention weight visualization
- Interactive attention heatmaps overlaid on images
- Patch-level attention for understanding visual focus

**3. Direct Logit Attribution**
- Analyze how different image patches contribute to specific token predictions
- Understand the relationship between visual features and model outputs
- Interactive attribution maps

#### Web Interface Usage

1. **Upload an image** using the file upload area
2. **Specify a task** (e.g., "Click on the search button")
3. **Optional: Add assistant prefill** to guide the response format
4. **Choose analysis type**:
   - **Predict**: Generate full response with coordinates
   - **Forward Pass**: Analyze next-token predictions
   - **Attention**: Visualize attention patterns for specific layers/heads
   - **Attribution**: See how image patches influence predictions

#### API Endpoints

The web app exposes REST API endpoints for programmatic access:

```bash
# Upload image
POST /api/upload
Content-Type: multipart/form-data

# Run prediction
POST /api/predict
{
  "filename": "uploaded_image.png",
  "task": "Click on the login button",
  "assistant_prefill": ""
}

# Forward pass analysis
POST /api/forward
{
  "filename": "uploaded_image.png",
  "task": "Describe the interface"
}

# Attention analysis
POST /api/attention
{
  "filename": "uploaded_image.png",
  "task": "Click here",
  "layer": 15,
  "head": 8
}
```

### Model Manager

The web app uses a `ModelManager` class that handles:
- **Background model loading** to avoid blocking the interface
- **Thread-safe model access** for concurrent requests
- **Coordinate parsing** from model outputs
- **Error handling** and status reporting
- **Caching** for improved performance

## Development

### Running Tests
```bash
# Run all tests
uv run pytest

# Run specific test categories
uv run pytest tests/hookedvlm_test.py      # Core model tests
uv run pytest tests/test_batching.py       # Batch processing tests
uv run pytest tests/test_prefill_functionality.py  # Prefill tests
```

### Code Quality
```bash
# Lint and format code
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Architecture

- **HookedVLM** (`src/vlm_spectra/core/hooked_vlm.py`): Main model wrapper with interpretability hooks
- **ModelAdapter** (`src/vlm_spectra/models/ModelAdapter.py`): Adapter pattern for different model architectures
- **Logit Lens** (`src/vlm_spectra/logit_lens/`): Visualization tools for token predictions across layers
- **Web App** (`src/vlm_spectra/web_app/`): Flask-based interface for interactive analysis
- **Utilities** (`src/vlm_spectra/utils/`): Image preprocessing and model-specific utilities

## Research Applications

This toolkit is designed for interpretability research on vision-language models, particularly:
- Understanding how VLMs process visual information at different layers
- Analyzing attention patterns between visual and textual modalities
- Investigating the relationship between image patches and token predictions
- Developing better GUI automation systems through model introspection

## License

MIT
