import os
import re
import time
import threading
from typing import Dict, Tuple, Any, Optional
import torch
import numpy as np

from vlm_spectra import HookedVLM
from vlm_spectra.models.registry import ModelRegistry

class ModelManager:
    """
    Manages VLM model loading and provides analysis methods for the web application.

    This class handles model initialization, caching, and provides various analysis
    methods including prediction, forward pass analysis, attention analysis, and
    direct logit attribution.
    """

    def __init__(self):
        self.model = None
        self.is_loading = False
        self.is_ready = False
        self.error_message = None
        self.load_lock = threading.Lock()
        self.model_options = {
            "ui-tars": {
                "label": "UI-TARS 1.5 7B",
                "model_name": "ByteDance-Seed/UI-TARS-1.5-7B",
            },
            "qwen3-vl": {
                "label": "Qwen3-VL 8B Instruct",
                "model_name": "Qwen/Qwen3-VL-8B-Instruct",
            },
        }
        self.current_model_id = "qwen3-vl"
        self.pending_model_id = None

    def get_model_options(self) -> Dict[str, Any]:
        supported_models = set(ModelRegistry.list_supported_models())
        options = []
        for model_id, info in self.model_options.items():
            options.append(
                {
                    "id": model_id,
                    "label": info["label"],
                    "model_name": info["model_name"],
                    "available": info["model_name"] in supported_models,
                }
            )
        return {
            "options": options,
            "active_model_id": self.current_model_id,
        }

    def get_current_model_label(self) -> str:
        return self.model_options[self.current_model_id]["label"]

    def get_current_model_name(self) -> str:
        return self.model_options[self.current_model_id]["model_name"]

    def is_model_available(self, model_id: str) -> bool:
        supported_models = set(ModelRegistry.list_supported_models())
        return self.model_options[model_id]["model_name"] in supported_models

    def get_patch_info(self) -> Dict[str, Any]:
        """
        Get patch size information for the currently loaded model.

        Returns:
            Dictionary containing patch_size, spatial_merge_size, effective_patch_size, and model_id
        """
        if not self.is_ready or self.model is None:
            return {
                'error': 'Model not ready',
                'patch_size': 14,
                'spatial_merge_size': 2,
                'effective_patch_size': 28,
                'model_id': self.current_model_id
            }

        from vlm_spectra.preprocessing.utils.vision_info import resolve_patch_params

        vision_config = self.model.model.config.vision_config
        model_name = self.model_options[self.current_model_id]["model_name"]
        patch_size, spatial_merge_size = resolve_patch_params(vision_config, model_name)
        effective_patch_size = patch_size * spatial_merge_size

        return {
            'patch_size': patch_size,
            'spatial_merge_size': spatial_merge_size,
            'effective_patch_size': effective_patch_size,
            'model_id': self.current_model_id
        }

    def generate_square_image(
        self,
        grid_colors: list,
        grid_rows: int,
        grid_cols: int
    ) -> Dict[str, Any]:
        """
        Generate an image from a grid of colors for model analysis.

        Each cell in the grid corresponds to one model patch. The image is saved
        to the uploads folder and can be used with existing analysis endpoints.

        Args:
            grid_colors: 2D list of color strings (e.g., [['red', 'blue'], ['black', 'white']])
            grid_rows: Number of rows in the grid
            grid_cols: Number of columns in the grid

        Returns:
            Dictionary containing success status, filename, url, dimensions, and patch info
        """
        from PIL import Image

        # Get effective patch size from current model
        patch_info = self.get_patch_info()
        effective_patch_size = patch_info['effective_patch_size']

        # Color mapping
        COLOR_MAP = {
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'yellow': (255, 255, 0),
            'cyan': (0, 255, 255),
            'magenta': (255, 0, 255),
        }

        # Calculate image dimensions
        img_width = grid_cols * effective_patch_size
        img_height = grid_rows * effective_patch_size

        # Create image with black background
        image = Image.new('RGB', (img_width, img_height), (0, 0, 0))

        # Fill patches
        for row in range(grid_rows):
            for col in range(grid_cols):
                if row < len(grid_colors) and col < len(grid_colors[row]):
                    color_name = grid_colors[row][col]
                else:
                    color_name = 'black'

                rgb = COLOR_MAP.get(color_name, (0, 0, 0))

                x_start = col * effective_patch_size
                y_start = row * effective_patch_size

                # Fill the patch region
                for y in range(y_start, y_start + effective_patch_size):
                    for x in range(x_start, x_start + effective_patch_size):
                        image.putpixel((x, y), rgb)

        # Save to uploads folder
        timestamp = int(time.time() * 1000)
        filename = f"generated_{timestamp}_{grid_rows}x{grid_cols}.png"

        script_dir = os.path.dirname(os.path.abspath(__file__))
        upload_folder = os.path.join(script_dir, 'static', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, filename)

        image.save(filepath, 'PNG')

        return {
            'success': True,
            'filename': filename,
            'url': f'/static/uploads/{filename}',
            'dimensions': {'width': img_width, 'height': img_height},
            'grid_size': {'rows': grid_rows, 'cols': grid_cols},
            'patch_size': effective_patch_size
        }

    def start_model_loading(self, model_id: Optional[str] = None) -> str:
        """Start loading a model in a background thread."""
        if model_id is None:
            model_id = self.current_model_id
        if model_id not in self.model_options:
            raise ValueError(f"Unknown model id: {model_id}")
        if not self.is_model_available(model_id):
            return "unavailable"

        with self.load_lock:
            if self.is_loading:
                self.pending_model_id = model_id
                return "queued"

            if self.is_ready and self.current_model_id == model_id:
                return "ready"

            self.pending_model_id = None
            self.current_model_id = model_id
            self.is_ready = False
            self.error_message = None
            self.is_loading = True

            model_thread = threading.Thread(target=self.load_model, args=(model_id,))
            model_thread.daemon = True
            model_thread.start()
            return "loading"
        
    def _release_model(self) -> None:
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(self, model_id: Optional[str] = None):
        """Load the model in a background thread"""
        if model_id is None:
            model_id = self.current_model_id

        model_name = self.model_options[model_id]["model_name"]
        model_label = self.model_options[model_id]["label"]

        try:
            print(f"Loading HookedVLM model: {model_label} ({model_name})")
            self._release_model()
            self.model = HookedVLM.from_pretrained(model_name=model_name, device="auto")
            self.is_ready = True
            print("Model loaded successfully!")
        except Exception as e:
            self.model = None
            self.error_message = str(e)
            self.is_ready = False
            print(f"Error loading model: {e}")
        finally:
            next_model_id = None
            with self.load_lock:
                self.is_loading = False
                if self.pending_model_id and self.pending_model_id != model_id:
                    next_model_id = self.pending_model_id
                    self.pending_model_id = None

            if next_model_id:
                self.start_model_loading(next_model_id)
    
    def parse_coordinates(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Parse coordinates from model output text containing box notation.

        Args:
            text: Model output text potentially containing box coordinates

        Returns:
            Tuple of (x, y) coordinates if found, (None, None) otherwise
        """
        box_pattern = r"<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>"
        match = re.search(box_pattern, text)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None
    
    def predict_from_image(
        self,
        image_path: str,
        task: str,
        assistant_prefill: str = ""
    ) -> Dict[str, Any]:
        """
        Run model prediction on uploaded image.

        Args:
            image_path: Path to the image file
            task: Task description for the model
            assistant_prefill: Optional prefill text for the assistant response

        Returns:
            Dictionary containing prediction results, inference time, and success status
        """
        if not self.is_ready:
            raise RuntimeError("Model not ready")
            
        try:
            from PIL import Image
            
            # Load the uploaded image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare model inputs
            inputs = self.model.prepare_messages(task, image, assistant_prefill=assistant_prefill)
            
            # Ensure inputs are on correct device
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(self.model.device)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if torch.is_tensor(subvalue):
                            value[subkey] = subvalue.to(self.model.device)
            
            # Generate model output
            start_time = time.time()
            outputs = self.model.generate(inputs)
            inference_time = time.time() - start_time
            
            # Decode output
            output_text = self.model.processor.tokenizer.decode(
                outputs.sequences[0], skip_special_tokens=False
            )
            
            # Parse predicted coordinates
            pred_x, pred_y = self.parse_coordinates(output_text)
            
            # Clean up output text for display
            clean_output = output_text.split("assistant\n")[-1] if "assistant\n" in output_text else output_text

            # Extract filename from path and create proper URL
            filename = os.path.basename(image_path)

            return {
                'success': True,
                'image_url': f'/static/uploads/{filename}',
                'task': task,
                'prefill': assistant_prefill,
                'prediction': {
                    'x': pred_x,
                    'y': pred_y
                },
                'output_text': clean_output,
                'inference_time': round(inference_time, 2),
                'image_size': image.size
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def direct_logit_attribution(
        self,
        image_path: str,
        task: str,
        assistant_prefill: str = ""
    ) -> Dict[str, Any]:
        """Run direct logit attribution analysis on uploaded image"""
        if not self.is_ready:
            raise RuntimeError("Model not ready")
            
        try:
            from PIL import Image
            import torch.nn.functional as F
            
            # Load the uploaded image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare model inputs
            inputs = self.model.prepare_messages(task, image, assistant_prefill=assistant_prefill)
            
            # Ensure inputs are on correct device
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(self.model.device)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if torch.is_tensor(subvalue):
                            value[subkey] = subvalue.to(self.model.device)
            
            # Run forward pass with cache to get attention heads, MLP outputs,
            # and the final residual stream (needed for frozen norm in DLA)
            last_layer = self.model.adapter.lm_num_layers - 1
            start_time = time.time()
            with self.model.run_with_cache([
                "lm.blocks.*.attn.hook_head_out",
                "lm.blocks.*.mlp.hook_out",
                f"lm.blocks.{last_layer}.hook_resid_post",
            ]):
                outputs = self.model.forward(inputs)
            inference_time = time.time() - start_time
            
            # Extract logits from the last token position
            logits = outputs.logits[0, -1, :]  # [vocab_size]
            
            # Get probabilities and top 10 tokens
            probs = F.softmax(logits, dim=-1)
            top_probs, top_indices = torch.topk(probs, k=10)
            top_logits = logits[top_indices]
            
            # Convert to token strings
            tokenizer = self.model.processor.tokenizer
            top_tokens = []
            
            for i in range(10):
                token_id = top_indices[i].item()
                token_str = tokenizer.decode([token_id])
                
                top_tokens.append({
                    'token': token_str,
                    'token_id': token_id,
                    'probability': top_probs[i].item(),
                    'logit': top_logits[i].item()
                })
            
            # Extract attention heads and MLP outputs from cache
            attention_heads = {}
            mlp_outputs = {}
            
            for layer_idx in range(self.model.adapter.lm_num_layers):
                # Per-head contributions: shape (batch_size, seq_len, num_heads, hidden_dim)
                attn_key = f"lm.blocks.{layer_idx}.attn.hook_head_out"
                if attn_key in self.model.cache:
                    attention_heads[layer_idx] = self.model.cache[attn_key][0]  # Remove batch dimension

                # MLP outputs: shape (batch_size, seq_len, hidden_dim)
                mlp_key = f"lm.blocks.{layer_idx}.mlp.hook_out"
                if mlp_key in self.model.cache:
                    mlp_outputs[layer_idx] = self.model.cache[mlp_key][0]  # Remove batch dimension
            
            # Extract the final residual stream at last token position for frozen norm
            resid_key = f"lm.blocks.{last_layer}.hook_resid_post"
            residual_stream = self.model.cache[resid_key][0, -1, :]  # [hidden_dim]

            # Compute DLA for all top 10 tokens
            dla_results = self._compute_dla_for_tokens(
                attention_heads, mlp_outputs,
                [token['token_id'] for token in top_tokens],
                residual_stream,
            )

            # Extract filename from path and create proper URL
            filename = os.path.basename(image_path)

            return {
                'success': True,
                'image_url': f'/static/uploads/{filename}',
                'task': task,
                'prefill': assistant_prefill,
                'top_tokens': top_tokens,
                'dla_data': dla_results,
                'token_position': f"Position {inputs['input_ids'].shape[1]}",
                'inference_time': round(inference_time, 2),
                'image_size': image.size
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _compute_dla_for_tokens(self, attention_heads, mlp_outputs, token_ids, residual_stream):
        """Compute direct logit attribution for given tokens using frozen norm.

        Uses the "linearized LayerNorm" approach: the RMS scale factor is
        computed once from the full residual stream and applied uniformly to
        all components.  This gives a valid linear decomposition where
        individual contributions sum to the logit (embedding baseline excluded).
        """
        # Get the unembedding matrix (lm_head) and normalization layer
        lm_head = self.model.model.lm_head
        norm_layer = self.model.adapter.lm_norm
        W_U = lm_head.weight  # Shape: (vocab_size, hidden_dim)

        norm_device = norm_layer.weight.device
        lm_head_device = W_U.device

        # --- Frozen norm: compute shared RMS scale from full residual stream ---
        # RMSNorm(x) = gamma * x / sqrt(mean(x^2) + eps)
        # We freeze the denominator using the full residual stream so the
        # decomposition is linear: contribution_i = (gamma / rms) * comp_i @ W_U^T
        eps = getattr(norm_layer, 'variance_epsilon', getattr(norm_layer, 'eps', 1e-6))
        residual = residual_stream.to(norm_device).to(torch.float32)
        rms_scale = torch.rsqrt(residual.pow(2).mean(-1) + eps)  # scalar
        frozen_scale = (norm_layer.weight * rms_scale).to(lm_head_device)  # [hidden_dim]

        # Index only the target rows from W_U first, then scale — avoids
        # materializing the full [vocab_size, hidden_dim] scaled matrix.
        token_id_tensor = torch.tensor(token_ids, device=lm_head_device, dtype=torch.long)
        W_U_target = (W_U[token_id_tensor, :] * frozen_scale.unsqueeze(0)).to(W_U.dtype)  # [num_tokens, hidden_dim]

        num_layers = self.model.adapter.lm_num_layers
        if attention_heads:
            first_layer = sorted(attention_heads.keys())[0]
            num_heads = attention_heads[first_layer].shape[1]
        else:
            num_heads = 0
        num_tokens = len(token_ids)

        # Initialize results arrays
        head_contributions = np.zeros((num_layers, num_heads, num_tokens))
        layer_att_contributions = np.zeros((num_layers, num_tokens))
        mlp_contributions = np.zeros((num_layers, num_tokens))

        # Compute contributions for each layer
        for layer_idx in range(num_layers):
            # Process attention heads
            if layer_idx in attention_heads:
                # All heads at once: [num_heads, hidden_dim] (last token position)
                all_heads = attention_heads[layer_idx][-1, :, :].to(lm_head_device)  # [num_heads, hidden_dim]

                # Compute logit contributions for all heads and target tokens at once
                # [num_heads, hidden_dim] @ [hidden_dim, num_tokens] -> [num_heads, num_tokens]
                head_logits = all_heads @ W_U_target.T
                head_contributions[layer_idx] = head_logits.float().cpu().detach().numpy()
                layer_att_contributions[layer_idx] = head_logits.sum(dim=0).float().cpu().detach().numpy()

            # Process MLP outputs
            if layer_idx in mlp_outputs:
                mlp_output = mlp_outputs[layer_idx][-1, :].to(lm_head_device)  # [hidden_dim]
                mlp_logits = mlp_output @ W_U_target.T  # [num_tokens]
                mlp_contributions[layer_idx] = mlp_logits.float().cpu().detach().numpy()

        return {
            'head_contributions': head_contributions.tolist(),  # [layers, heads, tokens]
            'layer_att_contributions': layer_att_contributions.tolist(),  # [layers, tokens]
            'mlp_contributions': mlp_contributions.tolist(),  # [layers, tokens]
            'num_layers': num_layers,
            'num_heads': num_heads
        }

    def attention_analysis(
        self,
        image_path: str,
        task: str,
        layer: int,
        head: int,
        assistant_prefill: str = ""
    ) -> Dict[str, Any]:
        """Run attention analysis for a specific layer and head"""
        if not self.is_ready:
            raise RuntimeError("Model not ready")
            
        try:
            from PIL import Image

            # Load the uploaded image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Prepare model inputs and get text for token analysis
            inputs, full_text = self.model.prepare_messages(task, image, assistant_prefill=assistant_prefill, return_text=True)

            # Ensure inputs are on correct device
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(self.model.device)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if torch.is_tensor(subvalue):
                            value[subkey] = subvalue.to(self.model.device)

            # Run forward pass with attention pattern cache
            start_time = time.time()
            with self.model.run_with_cache(["lm.blocks.*.attn.hook_pattern"]):
                self.model.forward(inputs)
            inference_time = time.time() - start_time

            # Extract attention patterns for the specified layer
            attention_key = f"lm.blocks.{layer}.attn.hook_pattern"
            if attention_key not in self.model.cache:
                raise ValueError(f"Attention patterns not found for layer {layer}")

            # Get attention weights: [batch, num_heads, seq_len, seq_len]
            attention_weights = self.model.cache[attention_key]
            batch_size, num_heads, seq_len, _ = attention_weights.shape
            
            # Validate head index
            if head >= num_heads:
                raise ValueError(f"Head {head} not available (model has {num_heads} heads)")
            
            # Extract attention for ALL heads and last token position (query)
            # Shape: [num_heads, seq_len] - attention weights from last token to all positions for all heads
            all_heads_attention = attention_weights[0, :, -1, :].float().cpu().numpy()
            
            # Also get the specific requested head for backward compatibility
            last_token_attention = all_heads_attention[head]
            
            # Get tokenizer and decode tokens for analysis
            tokenizer = self.model.processor.tokenizer
            input_ids = inputs['input_ids'][0].long().cpu().numpy()  # Remove batch dimension
            
            # Tokenize the full text to understand token boundaries
            tokens = []
            for token_id in input_ids:
                token_str = tokenizer.decode([token_id])
                tokens.append({
                    'id': int(token_id),
                    'text': token_str,
                    'attention': float(last_token_attention[len(tokens)]) if len(tokens) < len(last_token_attention) else 0.0
                })
            
            # Get the correct image token range using the existing method
            try:
                image_start_idx, image_end_idx = self.model.get_image_token_range(inputs)
            except ValueError:
                # If no image tokens found, return error
                return {
                    'success': False,
                    'error': 'No image tokens found in the input sequence'
                }
            
            # Calculate image patch information using existing model logic
            # Get vision config for patch information
            vision_config = self.model.model.config.vision_config
            model_name = getattr(self.model, "model_name", None)
            from vlm_spectra.preprocessing.utils.vision_info import (
                resolve_patch_params,
            )
            patch_size, spatial_merge_size = resolve_patch_params(
                vision_config, model_name
            )
            
            # Calculate resized dimensions using the model's logic
            width, height = image.size
            from vlm_spectra.preprocessing.utils.vision_info import (
                smart_resize,
                MIN_PIXELS,
                MAX_PIXELS,
            )
            resize_factor = patch_size * spatial_merge_size
            resized_height, resized_width = smart_resize(
                height,
                width,
                factor=resize_factor,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS,
            )
            
            # Calculate grid dimensions
            grid_h = resized_height // patch_size
            grid_w = resized_width // patch_size
            
            # After spatial merge
            merged_grid_h = grid_h // spatial_merge_size
            merged_grid_w = grid_w // spatial_merge_size
            effective_patch_size = patch_size * spatial_merge_size
            
            # Calculate number of image patches
            num_image_patches = merged_grid_h * merged_grid_w
            
            # Extract attention for image patches using correct range
            # image_start_idx and image_end_idx are inclusive, so we need image_end_idx + 1 for slicing
            image_patch_attention = last_token_attention[image_start_idx:image_end_idx + 1]
            
            # Ensure we have the expected number of image patches
            if len(image_patch_attention) != num_image_patches:
                print(f"Warning: Expected {num_image_patches} image patches, but got {len(image_patch_attention)} from range [{image_start_idx}:{image_end_idx + 1}]")
                # Pad or truncate as needed
                if len(image_patch_attention) < num_image_patches:
                    # Pad with zeros if we have fewer patches than expected
                    image_patch_attention = list(image_patch_attention) + [0.0] * (num_image_patches - len(image_patch_attention))
                else:
                    # Truncate if we have more patches than expected
                    image_patch_attention = image_patch_attention[:num_image_patches]
            
            # Create patch coordinate mappings for the requested head (backward compatibility)
            patches = []
            for i in range(merged_grid_h):
                for j in range(merged_grid_w):
                    patch_idx = i * merged_grid_w + j
                    if patch_idx < len(image_patch_attention):
                        # Calculate patch coordinates in resized image space
                        x_start = j * effective_patch_size
                        y_start = i * effective_patch_size
                        x_end = min((j + 1) * effective_patch_size, resized_width)
                        y_end = min((i + 1) * effective_patch_size, resized_height)
                        
                        # Scale coordinates back to original image space
                        scale_x = width / resized_width
                        scale_y = height / resized_height
                        
                        patches.append({
                            'patch_idx': patch_idx,
                            'grid_pos': [i, j],
                            'bbox': [
                                int(x_start * scale_x),
                                int(y_start * scale_y),
                                int(x_end * scale_x),
                                int(y_end * scale_y)
                            ],
                            'attention': float(image_patch_attention[patch_idx])
                        })
            
            # Create patch data for ALL heads
            all_patches = {}
            for head_idx in range(num_heads):
                head_image_attention = all_heads_attention[head_idx][image_start_idx:image_end_idx + 1]
                
                # Ensure we have the expected number of image patches for this head
                if len(head_image_attention) != num_image_patches:
                    # Pad or truncate as needed
                    if len(head_image_attention) < num_image_patches:
                        head_image_attention = list(head_image_attention) + [0.0] * (num_image_patches - len(head_image_attention))
                    else:
                        head_image_attention = head_image_attention[:num_image_patches]
                
                head_patches = []
                for i in range(merged_grid_h):
                    for j in range(merged_grid_w):
                        patch_idx = i * merged_grid_w + j
                        if patch_idx < len(head_image_attention):
                            # Calculate patch coordinates in resized image space
                            x_start = j * effective_patch_size
                            y_start = i * effective_patch_size
                            x_end = min((j + 1) * effective_patch_size, resized_width)
                            y_end = min((i + 1) * effective_patch_size, resized_height)
                            
                            # Scale coordinates back to original image space
                            scale_x = width / resized_width
                            scale_y = height / resized_height
                            
                            head_patches.append({
                                'patch_idx': patch_idx,
                                'grid_pos': [i, j],
                                'bbox': [
                                    int(x_start * scale_x),
                                    int(y_start * scale_y),
                                    int(x_end * scale_x),
                                    int(y_end * scale_y)
                                ],
                                'attention': float(head_image_attention[patch_idx])
                            })
                
                all_patches[head_idx] = head_patches
            
            # Prepare text tokens with attention for ALL tokens in sequence (requested head)
            text_tokens = []
            
            # Get all tokens, marking image tokens specially
            for idx in range(len(tokens)):
                token_info = tokens[idx]
                attention_value = float(last_token_attention[idx]) if idx < len(last_token_attention) else 0.0
                
                # Determine if this is an image token
                is_image_token = image_start_idx <= idx <= image_end_idx
                
                # Calculate patch index if this is an image token
                patch_idx = None
                if is_image_token:
                    patch_idx = idx - image_start_idx
                
                text_tokens.append({
                    'token_id': token_info['id'],
                    'text': token_info['text'],
                    'attention': attention_value,
                    'position': idx,
                    'is_image_token': is_image_token,
                    'patch_idx': patch_idx
                })
            
            # Create text tokens data for ALL heads
            all_text_tokens = {}
            for head_idx in range(num_heads):
                head_attention = all_heads_attention[head_idx]
                head_text_tokens = []
                
                for idx in range(len(tokens)):
                    token_info = tokens[idx]
                    attention_value = float(head_attention[idx]) if idx < len(head_attention) else 0.0
                    
                    # Determine if this is an image token
                    is_image_token = image_start_idx <= idx <= image_end_idx
                    
                    # Calculate patch index if this is an image token
                    patch_idx = None
                    if is_image_token:
                        patch_idx = idx - image_start_idx
                    
                    head_text_tokens.append({
                        'token_id': token_info['id'],
                        'text': token_info['text'],
                        'attention': attention_value,
                        'position': idx,
                        'is_image_token': is_image_token,
                        'patch_idx': patch_idx
                    })
                
                all_text_tokens[head_idx] = head_text_tokens
            
            # Extract filename from path and create proper URL
            filename = os.path.basename(image_path)

            return {
                'success': True,
                'layer': layer,
                'head': head,
                'image_url': f'/static/uploads/{filename}',
                'task': task,
                'prefill': assistant_prefill,
                'attention_data': {
                    'patches': patches,
                    'text_tokens': text_tokens,
                    'full_attention': last_token_attention.tolist(),
                    'num_image_patches': num_image_patches,
                    'image_token_range': {
                        'start': image_start_idx,
                        'end': image_end_idx
                    },
                    'image_dimensions': {
                        'original': [width, height],
                        'resized': [resized_width, resized_height],
                        'grid': [merged_grid_h, merged_grid_w],
                        'patch_size': effective_patch_size
                    },
                    # NEW: All heads data for instant switching
                    'all_heads': {
                        'patches': all_patches,
                        'text_tokens': all_text_tokens,
                        'full_attention': all_heads_attention.tolist()
                    }
                },
                'token_position': f"Position {inputs['input_ids'].shape[1]} (last token)",
                'inference_time': round(inference_time, 2),
                'image_size': image.size,
                'model_info': {
                    'num_layers': self.model.adapter.lm_num_layers,
                    'num_heads': self.model.adapter.lm_num_heads
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def logit_lens_analysis(
        self,
        image_path: str,
        task: str,
        assistant_prefill: str = ""
    ) -> Dict[str, Any]:
        """Run logit lens analysis: top-k token predictions at every layer and position."""
        if not self.is_ready:
            raise RuntimeError("Model not ready")

        try:
            from PIL import Image
            from vlm_spectra.analysis.logit_lens import compute_logit_lens
            from vlm_spectra.preprocessing.utils.vision_info import (
                resolve_patch_params,
                smart_resize,
                MIN_PIXELS,
                MAX_PIXELS,
            )

            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')

            inputs = self.model.prepare_messages(task, image, assistant_prefill=assistant_prefill)

            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(self.model.device)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if torch.is_tensor(subvalue):
                            value[subkey] = subvalue.to(self.model.device)

            # Forward pass caching all layer residual streams
            start_time = time.time()
            with self.model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
                self.model.forward(inputs)
            inference_time = time.time() - start_time

            # Stack hidden states into tuple for compute_logit_lens
            num_layers = self.model.adapter.lm_num_layers
            hidden_states = []
            for layer_idx in range(num_layers):
                cache_key = f"lm.blocks.{layer_idx}.hook_resid_post"
                if cache_key in self.model.cache:
                    hidden_states.append(self.model.cache[cache_key])
            hidden_states = tuple(hidden_states)

            # Build token labels
            tokenizer = self.model.processor.tokenizer
            input_ids = inputs['input_ids'][0].long().cpu().numpy()

            try:
                image_start_idx, image_end_idx = self.model.get_image_token_range(inputs)
            except ValueError:
                image_start_idx, image_end_idx = -1, -1

            token_labels = []
            img_counter = 0
            for idx, tid in enumerate(input_ids):
                if image_start_idx <= idx <= image_end_idx:
                    token_labels.append(f"<IMG{img_counter:03d}>")
                    img_counter += 1
                else:
                    token_labels.append(tokenizer.decode([tid]))

            # Compute logit lens
            norm_layer = self.model.adapter.lm_norm
            lm_head = self.model.adapter.lm_head
            top_k = 5
            all_top_tokens = compute_logit_lens(
                hidden_states, norm_layer, lm_head, tokenizer, top_k=top_k
            )

            # Compute vision grid info for image patch highlighting
            width, height = image.size
            vision_config = self.model.model.config.vision_config
            model_name = getattr(self.model, "model_name", None)
            patch_size, spatial_merge_size = resolve_patch_params(vision_config, model_name)
            resize_factor = patch_size * spatial_merge_size
            resized_height, resized_width = smart_resize(
                height, width, factor=resize_factor,
                min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS,
            )
            grid_h = resized_height // patch_size
            grid_w = resized_width // patch_size
            merged_grid_h = grid_h // spatial_merge_size
            merged_grid_w = grid_w // spatial_merge_size
            effective_patch_size = patch_size * spatial_merge_size

            filename = os.path.basename(image_path)

            return {
                'success': True,
                'all_top_tokens': all_top_tokens,
                'token_labels': token_labels,
                'num_layers': len(all_top_tokens),
                'num_positions': len(token_labels),
                'image_url': f'/static/uploads/{filename}',
                'image_token_range': {
                    'start': image_start_idx,
                    'end': image_end_idx,
                },
                'grid_info': {
                    'merged_grid_h': merged_grid_h,
                    'merged_grid_w': merged_grid_w,
                    'effective_patch_size': effective_patch_size,
                    'original_width': width,
                    'original_height': height,
                    'resized_width': resized_width,
                    'resized_height': resized_height,
                },
                'inference_time': round(inference_time, 2),
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def forward_pass_analysis(
        self,
        image_path: str,
        task: str,
        assistant_prefill: str = ""
    ) -> Dict[str, Any]:
        """Run forward pass analysis on uploaded image and return top token predictions and layer-wise probabilities"""
        if not self.is_ready:
            raise RuntimeError("Model not ready")
            
        try:
            from PIL import Image
            import torch.nn.functional as F
            
            # Load the uploaded image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare model inputs
            inputs = self.model.prepare_messages(task, image, assistant_prefill=assistant_prefill)
            
            # Ensure inputs are on correct device
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    inputs[key] = value.to(self.model.device)
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if torch.is_tensor(subvalue):
                            value[subkey] = subvalue.to(self.model.device)
            
            # Run forward pass with cache to get layer-wise hidden states
            start_time = time.time()
            with self.model.run_with_cache(["lm.blocks.*.hook_resid_post"]):
                outputs = self.model.forward(inputs)
            inference_time = time.time() - start_time
            
            # Extract logits from the last token position (final layer)
            logits = outputs.logits[0, -1, :]  # [vocab_size]
            
            # Get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Get top 10 tokens
            top_probs, top_indices = torch.topk(probs, k=10)
            top_logits = logits[top_indices]
            
            # Convert to token strings
            tokenizer = self.model.processor.tokenizer
            top_tokens = []
            
            for i in range(10):
                token_id = top_indices[i].item()
                token_str = tokenizer.decode([token_id])
                
                top_tokens.append({
                    'token': token_str,
                    'token_id': token_id,
                    'probability': top_probs[i].item(),
                    'logit': top_logits[i].item()
                })
            
            # Compute layer-wise probabilities for top 10 tokens
            layer_probabilities = []
            top_token_ids = [token['token_id'] for token in top_tokens]
            
            # Get language model head and normalization layer for computing logits from hidden states
            lm_head = self.model.adapter.lm_head
            norm_layer = self.model.adapter.lm_norm
            
            # Process each layer's hidden states
            for layer_idx in range(self.model.adapter.lm_num_layers):
                cache_key = f"lm.blocks.{layer_idx}.hook_resid_post"
                if cache_key in self.model.cache:
                    # Get hidden state for last token position at this layer
                    hidden_state = self.model.cache[cache_key][0, -1, :]  # [hidden_dim]
                    
                    # Apply RMSNorm before language model head (crucial step!)
                    # Ensure hidden_state is on the same device as norm_layer
                    hidden_state = hidden_state.to(norm_layer.weight.device)
                    normalized_hidden = norm_layer(hidden_state)  # [hidden_dim]
                    
                    # Compute logits for this layer using normalized hidden state
                    layer_logits = lm_head(normalized_hidden)  # [vocab_size]
                    
                    # Get probabilities
                    layer_probs = F.softmax(layer_logits, dim=-1)
                    
                    # Extract probabilities for our top 10 tokens
                    layer_token_probs = []
                    for token_id in top_token_ids:
                        layer_token_probs.append(layer_probs[token_id].item())
                    
                    layer_probabilities.append(layer_token_probs)

            # Extract filename from path and create proper URL
            filename = os.path.basename(image_path)

            return {
                'success': True,
                'image_url': f'/static/uploads/{filename}',
                'task': task,
                'prefill': assistant_prefill,
                'top_tokens': top_tokens,
                'layer_probabilities': layer_probabilities,  # New field: [num_layers][10] probabilities
                'token_position': f"Position {inputs['input_ids'].shape[1]}",
                'inference_time': round(inference_time, 2),
                'image_size': image.size,
                'model_info': {
                    'num_layers': self.model.adapter.lm_num_layers,
                    'num_heads': self.model.adapter.lm_num_heads
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
