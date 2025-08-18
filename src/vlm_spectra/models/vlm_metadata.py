from PIL import Image
from typing import Dict

class VLMMetadataExtractor:
    """Extract metadata needed for logit lens visualization from various VLMs"""
    
    @staticmethod
    def extract_metadata_qwen(
        model,
        processor,
        inputs: Dict,
        original_image: Image.Image
    ) -> Dict:
        """Extract metadata from Qwen model inputs/outputs"""
        from vlm_spectra.utils.qwen_25_vl_utils import smart_resize, IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS
        
        # Get image dimensions
        width, height = original_image.size
        
        # Calculate resized dimensions
        resized_height, resized_width = smart_resize(
            height, width,
            factor=IMAGE_FACTOR,
            min_pixels=MIN_PIXELS,
            max_pixels=MAX_PIXELS
        )
        
        # Get vision config
        vision_config = model.config.vision_config
        patch_size = vision_config.patch_size
        spatial_merge_size = vision_config.spatial_merge_size
        
        # Calculate grid dimensions
        grid_h = resized_height // patch_size
        grid_w = resized_width // patch_size
        
        # After spatial merge
        merged_grid_h = grid_h // spatial_merge_size
        merged_grid_w = grid_w // spatial_merge_size
        
        # Get token information
        input_ids = inputs['input_ids'].squeeze(0)
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        
        # Create token labels and identify image token positions
        token_labels = []
        image_token_positions = []
        img_token_counter = 0
        
        for i, token_id in enumerate(input_ids.tolist()):
            if token_id == image_token_id:
                token_labels.append(f"<IMG{(img_token_counter+1):03d}>")
                image_token_positions.append(i)
                img_token_counter += 1
            else:
                token_labels.append(processor.tokenizer.decode([token_id]))
        
        return {
            'token_labels': token_labels,
            'image_token_positions': image_token_positions,
            'image_size': (resized_width, resized_height),
            'grid_size': (merged_grid_h, merged_grid_w),
            'patch_size': patch_size * spatial_merge_size,  # Effective patch size
            'num_image_tokens': len(image_token_positions),
            'total_patches': merged_grid_h * merged_grid_w,
        }
    
    @staticmethod
    def extract_metadata_llava(
        model,
        processor,
        inputs: Dict,
        original_image: Image.Image,
        image_size: int = 336,
        patch_size: int = 14
    ) -> Dict:
        """Extract metadata from LLaVA model inputs/outputs"""
        # For LLaVA, images are resized to fixed size
        resized_size = (image_size, image_size)
        grid_size = image_size // patch_size
        
        # Get token information
        input_ids = inputs['input_ids'].squeeze(0)
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")  # or model.config.image_token_id
        
        # Create token labels
        token_labels = []
        image_token_positions = []
        
        # Find image token and expand it
        for i, token_id in enumerate(input_ids.tolist()):
            if token_id == image_token_id:
                # LLaVA replaces single image token with multiple patches
                num_patches = grid_size * grid_size
                for j in range(num_patches):
                    token_labels.append(f"<IMG{(j+1):03d}>")
                    image_token_positions.append(len(token_labels) - 1)
            else:
                token_labels.append(processor.tokenizer.decode([token_id]))
        
        return {
            'token_labels': token_labels,
            'image_token_positions': image_token_positions,
            'image_size': resized_size,
            'grid_size': (grid_size, grid_size),
            'patch_size': patch_size,
            'num_image_tokens': len(image_token_positions),
            'total_patches': grid_size * grid_size,
        }