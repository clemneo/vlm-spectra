import os
import re
import time
import threading
from typing import Dict, Tuple, Any, Optional
import torch

from vlm_spectra.models.HookedVLM import HookedVLM

class ModelManager:
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
                self.model = HookedVLM(model_name="ByteDance-Seed/UI-TARS-1.5-7B", device="auto")
                
                self.is_ready = True
                print("Model loaded successfully!")
                
            except Exception as e:
                self.error_message = str(e)
                print(f"Error loading model: {e}")
            finally:
                self.is_loading = False
    
    def parse_coordinates(self, text: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse coordinates from model output text"""
        box_pattern = r"<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>"
        match = re.search(box_pattern, text)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None
    
    def predict_from_image(
        self,
        image_path: str,
        task: str
    ) -> Dict[str, Any]:
        """Run model prediction on uploaded image"""
        if not self.is_ready:
            raise RuntimeError("Model not ready")
            
        try:
            from PIL import Image
            
            # Load the uploaded image
            image = Image.open(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Prepare model inputs
            inputs = self.model.prepare_messages(task, image)
            
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
            
            return {
                'success': True,
                'image_url': f'/{image_path}',
                'task': task,
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