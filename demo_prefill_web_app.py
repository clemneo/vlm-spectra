#!/usr/bin/env python3
"""
Demo script showing the web app prefill functionality.
This simulates how the web app would call the model_manager with prefill.
"""

from src.vlm_spectra.web_app.model_manager import ModelManager
from PIL import Image
import numpy as np
import os
import json

def create_demo_image():
    """Create a simple demo image"""
    # Create a simple red square on white background
    image = Image.new('RGB', (300, 200), 'white')
    pixels = image.load()
    
    # Draw a red square from (50, 50) to (100, 100)
    for x in range(50, 101):
        for y in range(50, 101):
            pixels[x, y] = (255, 0, 0)  # Red
    
    return image

def demo_prefill_functionality():
    """Demonstrate prefill functionality in the web app context"""
    print("🚀 VLM-Spectra Web App Prefill Demo")
    print("=" * 50)
    
    # Create demo image
    print("📸 Creating demo image...")
    demo_image = create_demo_image()
    image_path = "/tmp/demo_red_square.png"
    demo_image.save(image_path)
    print(f"   Saved to: {image_path}")
    
    # Initialize model manager
    print("\n🤖 Loading model...")
    model_manager = ModelManager()
    model_manager.load_model()
    
    if not model_manager.is_ready:
        print("❌ Model failed to load")
        return
    
    print("✅ Model loaded successfully!")
    
    # Test cases
    test_cases = [
        {
            "name": "Standard Response",
            "task": "Click on the red square",
            "prefill": ""
        },
        {
            "name": "JSON Prefill",
            "task": "Click on the red square and respond in JSON format",
            "prefill": '{"action": "'
        },
        {
            "name": "Descriptive Prefill",
            "task": "Describe what you see and then click on the red element",
            "prefill": "I can see a red square located at"
        }
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} prefill scenarios:")
    print("-" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print(f"   Task: {test_case['task']}")
        print(f"   Prefill: '{test_case['prefill']}'")
        
        # Test prediction (generation)
        print("   🎯 Testing generation...")
        result = model_manager.predict_from_image(
            image_path=image_path,
            task=test_case['task'],
            assistant_prefill=test_case['prefill']
        )
        
        if result['success']:
            print(f"   ✅ Generation successful!")
            print(f"      Prediction: {result['prediction']}")
            print(f"      Inference time: {result['inference_time']}s")
            print(f"      Output sample: {result['output_text'][:100]}...")
        else:
            print(f"   ❌ Generation failed: {result['error']}")
        
        # Test forward pass
        print("   🔍 Testing forward pass...")
        result = model_manager.forward_pass_analysis(
            image_path=image_path,
            task=test_case['task'],
            assistant_prefill=test_case['prefill']
        )
        
        if result['success']:
            print(f"   ✅ Forward pass successful!")
            print(f"      Top token: '{result['top_tokens'][0]['token']}'")
            print(f"      Token probability: {result['top_tokens'][0]['probability']:.3f}")
            print(f"      Inference time: {result['inference_time']}s")
        else:
            print(f"   ❌ Forward pass failed: {result['error']}")
    
    # Cleanup
    os.remove(image_path)
    print(f"\n🧹 Cleaned up demo image: {image_path}")
    
    print("\n🎉 Demo completed successfully!")
    print("\nThe web app now supports assistant response prefilling!")
    print("Users can:")
    print("  • Leave the prefill field empty for standard behavior")
    print("  • Use JSON prefill like '{\"action\": \"' for structured responses")
    print("  • Use descriptive prefill to guide the response style")

if __name__ == "__main__":
    demo_prefill_functionality()