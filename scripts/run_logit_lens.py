import os
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path

from vlm_spectra.models.HookedVLM import HookedVLM
from vlm_spectra.models.vlm_metadata import VLMMetadataExtractor
from vlm_spectra.logit_lens.create_logit_lens import create_logit_lens


def is_image_file(filename):
    valid_extensions = ('.jpeg', '.jpg', '.png', '.JPEG', '.JPG', '.PNG')
    return filename.lower().endswith(valid_extensions)


def process_images_qwen(image_folder, save_folder, task="Describe the image.", num_images=None):
    """Process images using Qwen model"""
    Path(save_folder).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = HookedVLM(model_name="ByteDance-Seed/UI-TARS-1.5-7B")
    
    # Get model components
    components = model.get_model_components()
    
    # Load images
    image_files = [f for f in os.listdir(image_folder) if is_image_file(f)]
    if num_images:
        image_files = image_files[:num_images]
    
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    
    # Process each image
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            image = Image.open(image_path)
            
            # Prepare inputs
            inputs = model._prepare_messages(task, image)
            
            # Run forward pass
            outputs = model.forward(task, image, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            
            # Extract metadata for this model
            metadata = VLMMetadataExtractor.extract_metadata_qwen(
                model.model,
                model.processor,
                inputs,
                image
            )
            
            # Create the full prompt
            full_prompt = f"{model.prompt.format(language='English', instruction=task)}"
            
            # Create logit lens visualization
            create_logit_lens(
                hidden_states=hidden_states,
                norm=components['norm'],
                lm_head=components['lm_head'],
                tokenizer=components['tokenizer'],
                image=image,
                token_labels=metadata['token_labels'],
                image_size=metadata['image_size'],
                grid_size=metadata['grid_size'],
                patch_size=metadata['patch_size'],
                model_name=model.model_name.split("/")[-1],
                image_filename=image_path,
                prompt=full_prompt,
                save_folder=save_folder,
                misc_text=f"Total patches: {metadata['total_patches']}"
            )
            
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Process images with logit lens visualization")
    parser.add_argument("--image_folder", required=True, help="Path to the folder containing images")
    parser.add_argument("--save_folder", required=True, help="Path to save the results")
    parser.add_argument("--task", default="Describe the image.", help="Task instruction for the model")
    parser.add_argument("--num_images", type=int, help="Number of images to process (optional)")
    parser.add_argument("--model", default="qwen", choices=["qwen", "llava"], help="Model type to use")
    
    args = parser.parse_args()
    
    if args.model == "qwen":
        process_images_qwen(args.image_folder, args.save_folder, args.task, args.num_images)
    else:
        # You can add process_images_llava here when you implement LLaVA support
        raise NotImplementedError(f"Model {args.model} not yet implemented")


if __name__ == "__main__":
    main()