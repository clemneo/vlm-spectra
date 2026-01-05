import os
import re
import time
import threading
from typing import Dict, Tuple, Any, Optional
import torch
import numpy as np

from vlm_spectra import HookedVLM

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
        
    def load_model(self):
        """Load the model in a background thread"""
        with self.load_lock:
            if self.is_loading or self.is_ready:
                return
                
            self.is_loading = True
            self.error_message = None
            
            try:
                print("Loading HookedVLM model...")
                self.model = HookedVLM.from_pretrained(
                    model_name="ByteDance-Seed/UI-TARS-1.5-7B", device="auto"
                )
                
                self.is_ready = True
                print("Model loaded successfully!")
                
            except Exception as e:
                self.error_message = str(e)
                print(f"Error loading model: {e}")
            finally:
                self.is_loading = False
    
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
            
            # Run forward pass with cache to get attention heads and MLP outputs
            start_time = time.time()
            with self.model.run_with_cache(["lm_attn_out", "lm_mlp_out"]):
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
                # Attention heads: shape (batch_size, seq_len, num_heads, head_dim)
                attn_key = ("lm_attn_out", layer_idx)
                if attn_key in self.model.cache:
                    attention_heads[layer_idx] = self.model.cache[attn_key][0]  # Remove batch dimension
                
                # MLP outputs: shape (batch_size, seq_len, hidden_dim)
                mlp_key = ("lm_mlp_out", layer_idx)
                if mlp_key in self.model.cache:
                    mlp_outputs[layer_idx] = self.model.cache[mlp_key][0]  # Remove batch dimension
            
            # Compute DLA for all top 10 tokens
            dla_results = self._compute_dla_for_tokens(attention_heads, mlp_outputs, [token['token_id'] for token in top_tokens])

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

    def _compute_dla_for_tokens(self, attention_heads, mlp_outputs, token_ids):
        """Compute direct logit attribution for given tokens"""
        # Get the unembedding matrix (lm_head) and normalization layer
        lm_head = self.model.model.lm_head
        norm_layer = self.model.adapter.lm_norm
        W_U = lm_head.weight  # Shape: (vocab_size, hidden_dim)
        
        # Determine the device from where the norm layer weights are located
        norm_device = norm_layer.weight.device
        lm_head_device = W_U.device
        
        num_layers = len(attention_heads)
        num_heads = attention_heads[0].shape[1] if num_layers > 0 else 0
        num_tokens = len(token_ids)
        
        # Initialize results arrays
        head_contributions = np.zeros((num_layers, num_heads, num_tokens))
        layer_att_contributions = np.zeros((num_layers, num_tokens))
        mlp_contributions = np.zeros((num_layers, num_tokens))
        
        # Compute contributions for each layer
        for layer_idx in range(num_layers):
            # Process attention heads
            if layer_idx in attention_heads:
                layer_head_contribs = np.zeros((num_heads, num_tokens))
                
                for head_idx in range(num_heads):
                    # Get head contribution at final token position
                    head_contribution = attention_heads[layer_idx][-1, head_idx, :]  # [hidden_dim]
                    
                    # Move head_contribution to norm_layer device for normalization
                    head_contribution_norm = head_contribution.to(norm_device)
                    
                    # Apply RMSNorm before language model head (crucial step!)
                    normalized_head_contribution = norm_layer(head_contribution_norm)  # [hidden_dim]
                    
                    # Move normalized output to lm_head device for logit computation
                    normalized_head_contribution = normalized_head_contribution.to(lm_head_device)
                    
                    # Compute logit contribution: normalized_head_contribution @ W_U^T
                    logit_contributions = normalized_head_contribution @ W_U.T  # [vocab_size]
                    
                    # Extract contributions for all tokens
                    for token_idx, token_id in enumerate(token_ids):
                        contribution = logit_contributions[token_id].item()
                        head_contributions[layer_idx, head_idx, token_idx] = contribution
                        layer_head_contribs[head_idx, token_idx] = contribution
                
                # Sum head contributions for layer attention contribution
                layer_att_contributions[layer_idx, :] = np.sum(layer_head_contribs, axis=0)
            
            # Process MLP outputs
            if layer_idx in mlp_outputs:
                # Get MLP output at final token position
                mlp_output = mlp_outputs[layer_idx][-1, :]  # [hidden_dim]
                
                # Move mlp_output to norm_layer device for normalization
                mlp_output_norm = mlp_output.to(norm_device)
                
                # Apply RMSNorm before language model head (crucial step!)
                normalized_mlp_output = norm_layer(mlp_output_norm)  # [hidden_dim]
                
                # Move normalized output to lm_head device for logit computation
                normalized_mlp_output = normalized_mlp_output.to(lm_head_device)
                
                # Compute logit contribution: normalized_mlp_output @ W_U^T
                logit_contributions = normalized_mlp_output @ W_U.T  # [vocab_size]
                
                # Extract contributions for all tokens
                for token_idx, token_id in enumerate(token_ids):
                    contribution = logit_contributions[token_id].item()
                    mlp_contributions[layer_idx, token_idx] = contribution
        
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
            with self.model.run_with_cache(["lm_attn_pattern"]):
                self.model.forward(inputs)
            inference_time = time.time() - start_time
            
            # Extract attention patterns for the specified layer
            attention_key = ("lm_attn_pattern", layer)
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
            patch_size = vision_config.patch_size
            spatial_merge_size = vision_config.spatial_merge_size
            
            # Calculate resized dimensions using the model's logic
            width, height = image.size
            from vlm_spectra.preprocessing.utils.vision_info import (
                smart_resize,
                IMAGE_FACTOR,
                MIN_PIXELS,
                MAX_PIXELS,
            )
            resized_height, resized_width = smart_resize(
                height, width,
                factor=IMAGE_FACTOR,
                min_pixels=MIN_PIXELS,
                max_pixels=MAX_PIXELS
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
            with self.model.run_with_cache(["lm_resid_post"]):
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
                cache_key = ("lm_resid_post", layer_idx)
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
