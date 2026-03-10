import torch
from transformers import Swin2SRImageProcessor
from PIL import Image
import os
import numpy as np

# Import your model
from src.models.deblur_model import get_pretrained_deblur_model

def deblur_image(image_path, output_path, checkpoint_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running inference on: {device}")

    # 1. Load the model
    model = get_pretrained_deblur_model()
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"[*] Loaded fine-tuned weights from {checkpoint_path}")
    else:
        print("[!] Checkpoint not found. Proceeding with base Hugging Face weights.")
    
    model.to(device)
    model.eval()

    # 2. Prepare the blurry image
    print(f"[*] Processing {image_path}...")
    processor = Swin2SRImageProcessor()
    
    # NEW: Resize the image to match our training dimensions and save VRAM!
    # NEW: Smart Resize that preserves aspect ratio and saves VRAM!
    image = Image.open(image_path).convert("RGB")
    
    # Calculate new dimensions keeping the aspect ratio (max 384 pixels to be safe for 4GB)
    max_size = 384
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(max_size * (height / width))
    else:
        new_height = max_size
        new_width = int(max_size * (width / height))
        
    # Swin Transformers prefer dimensions that are multiples of 8
    new_width = new_width - (new_width % 8)
    new_height = new_height - (new_height % 8)
    
    image = image.resize((new_width, new_height))
    
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)

    # 3. Perform the Forward Pass
    with torch.no_grad():
        outputs = model(pixel_values)
        
    # 4. Post-process and save
    output_tensor = outputs.reconstruction.squeeze(0).cpu().clamp(0, 1)
    output_array = (output_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    restored_image = Image.fromarray(output_array)
    
    restored_image.save(output_path)
    print(f"[*] Success! Restored image saved to: {output_path}")

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    
    # Ensure this matches an actual file in your blur_docs folder!
    TEST_IMAGE_PATH = 'data/blur_docs/X51008164969.jpg' 
    OUTPUT_IMAGE_PATH = 'results/restored_test3.jpg'
    
    # Pointing to the final trained weights
    CHECKPOINT_PATH = 'checkpoints/lens_audit_deblur_ep10.pth'
    
    if os.path.exists(TEST_IMAGE_PATH):
        deblur_image(TEST_IMAGE_PATH, OUTPUT_IMAGE_PATH, CHECKPOINT_PATH)
    else:
        print(f"[!] Error: Could not find a test image at {TEST_IMAGE_PATH}")