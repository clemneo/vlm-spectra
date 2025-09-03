# Computer Vision Interpretability Demo

Interactive web application for demonstrating square generation and UI-TARS model inference.

## Features

- **Interactive Square Generation**: Configure grid size, square position, colors, and padding
- **Real-time Model Inference**: Uses UI-TARS-1.5-7B model for GUI automation tasks
- **Visual Results**: Display generated images with ground truth vs predicted coordinates
- **Performance Metrics**: Shows accuracy, coordinate distance, and inference time

## Setup

We use uv, so dependencies are as simple as:
1. Make sure vlm-spectra is your current working directory:
```bash
cd vlm-spectra
```
2. Run the demo server:
```bash
uv run src/vlm-spectra/web_app/app.py
```

3. Open your browser to: http://localhost:55556

## Usage

1. **Wait for Model Loading**: The model loads in the background (may take a few minutes)
2. **Upload image**: You can upload an image, or paste a screenshot

3. **Predict**: Click the button to run inference

4. **View Results**:
   - Image with coordinate overlays
   - Model's raw output text
   - Accuracy comparison (ground truth vs prediction)
   - Performance metrics

## Architecture

- **Flask Backend**: Lightweight web server on port 55556
- **Model Manager**: Background loading of HookedVLM with UI-TARS-1.5-7B
- **Square Generator**: Uses existing SquareGenerator class for image creation
- **Interactive Frontend**: Bootstrap UI with real-time updates

## Files

- `app.py`: Main Flask application with API endpoints
- `model_manager.py`: Model loading and inference logic
- `templates/index.html`: Web interface template
- `static/css/style.css`: Custom styling
- `static/js/app.js`: Frontend JavaScript functionality
- `static/images/`: Generated images storage

## API Endpoints

- `GET /`: Main demo interface
- `GET /api/model/status`: Model loading status
- `POST /api/generate`: Generate image and run prediction
- `GET /api/colors`: Available color options

## Requirements

- Python 3.10+
- CUDA-compatible GPU (for UI-TARS model)
- Dependencies from computer-interp and vlm-spectra projects