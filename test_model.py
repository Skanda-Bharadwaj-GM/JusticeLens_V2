import torch
import numpy as np
import time
from PIL import Image
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
HF_USERNAME = "skandab17" 
REPO_ID = f"{HF_USERNAME}/justice-lens-weights"
WEIGHT_FILE = "lens_cloud_latest_checkpoint.pth" 

INPUT_IMAGE = "new.png"  
OUTPUT_IMAGE = "restored_ocr_clear8.jpg" 
MAX_IMAGE_SIZE = 512 # Prevents out-of-memory errors by capping image size

def main():
    # 1. Hardware Detection
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[*] Hardware: NVIDIA GPU (CUDA) detected. Fast processing enabled!")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[*] Hardware: Apple Silicon (MPS) detected. Fast processing enabled!")
    else:
        device = torch.device("cpu")
        print("[!] WARNING: No GPU detected. Using CPU. The analysis step will take a while.")

    # 2. Download/Locate Weights
    print(f"\n[*] Fetching {WEIGHT_FILE} from Hugging Face...")
    try:
        weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHT_FILE)
    except Exception as e:
        print(f"[ERROR] Could not find the file. Ensure the repo and filename are correct.\n{e}")
        return

    # 3. Load Model Brain
    print("[*] Loading the base Swin2SR architecture...")
    processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
    model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

    # 4. Inject Custom Weights & Move to Hardware
    print("[*] Injecting Justice Lens weights and moving to hardware...")
    custom_weights = torch.load(weights_path, map_location=device)
    model.load_state_dict(custom_weights)
    model.to(device)
    model.eval() 

    # 5. Load and Prep Image
    print(f"\n[*] Preparing {INPUT_IMAGE}...")
    image = Image.open(INPUT_IMAGE).convert("RGB")
    
    # Smart resize to prevent computer freezing
    if max(image.size) > MAX_IMAGE_SIZE:
        print(f"[*] Image is very large {image.size}. Shrinking to safe test size...")
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        print(f"[*] New testing size: {image.size}")

    inputs = processor(images=image, return_tensors="pt").to(device)

    # 6. Run Inference
    print("\n[*] Analyzing image... (Please wait)")
    start_time = time.time()
    
    with torch.no_grad(): # Saves memory
        outputs = model(**inputs)
        
    end_time = time.time()
    print(f"[*] Analysis complete! It took {end_time - start_time:.2f} seconds.")

    # 7. Reconstruct and Save
    print("[*] Reconstructing the clear image...")
    output_tensor = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_tensor = np.transpose(output_tensor, (1, 2, 0))
    output_image = (output_tensor * 255.0).round().astype(np.uint8)
    
    restored_img = Image.fromarray(output_image)
    restored_img.save(OUTPUT_IMAGE)
    
    print(f"\n[SUCCESS] The deblurred image has been saved as: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()