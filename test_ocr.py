import os
import torch
import numpy as np
from PIL import Image
import pytesseract
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
HF_TOKEN = os.environ.get("HF_TOKEN") # Or paste your token as a string here temporarily
REPO_ID = "skandab17/justice-lens-weights"

# We are grabbing the absolute final save state before the credits ran out
CHECKPOINT_FILE = "lens_cloud_latest_checkpoint.pth" 
TEST_IMAGE = "new.png"
OUTPUT_IMAGE = "restored_ocr_clear8.jpg"

# Windows users ONLY: Uncomment and update this path if pytesseract throws a "Not Found" error
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# ---------------------

def main():
    print("[*] Initializing Justice Lens OCR Testing Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")

    # 1. Download Your Custom Brain
    print(f"[*] Fetching final weights ({CHECKPOINT_FILE}) from Hugging Face...")
    weights_path = hf_hub_download(repo_id=REPO_ID, filename=CHECKPOINT_FILE, token=HF_TOKEN)

    # 2. Load Base Model and Inject Your Weights
    print("[*] Loading base Swin2SR model and injecting custom weights...")
    processor = Swin2SRImageProcessor()
    model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
    
    # We use strict=False because we only trained a subset of the parameters
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()

    # 3. Load and Preprocess the Blurry Image
    print(f"[*] Loading test image: {TEST_IMAGE}")
    blurry_img = Image.open(TEST_IMAGE).convert("RGB")
    inputs = processor(images=blurry_img, return_tensors="pt").to(device)

    # 4. Run the AI Inference
    print("[*] Reconstructing document... (This might take a moment on CPU)")
    with torch.no_grad():
        outputs = model(pixel_values=inputs.pixel_values)
    
    # 5. Convert AI Output back to a standard Image
    reconstruction = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    reconstruction = np.moveaxis(reconstruction, 0, -1)
    reconstruction = (reconstruction * 255.0).round().astype(np.uint8)
    restored_img = Image.fromarray(reconstruction)
    
    # Save the visual result
    restored_img.save(OUTPUT_IMAGE)
    print(f"[*] Restored image saved locally as: {OUTPUT_IMAGE}")

    # 6. The Ultimate Test: OCR Comparison
    print("\n" + "="*50)
    print(" OCR PERFORMANCE REPORT")
    print("="*50)
    
    print("\n[SCENE 1: Extracting text from the ORIGINAL BLURRY image...]")
    blurry_text = pytesseract.image_to_string(blurry_img).strip()
    print(f"RESULT:\n{blurry_text if blurry_text else '<NO TEXT DETECTED BY TESSERACT>'}")
    
    print("\n[SCENE 2: Extracting text from the AI RESTORED image...]")
    restored_text = pytesseract.image_to_string(restored_img).strip()
    print(f"RESULT:\n{restored_text if restored_text else '<NO TEXT DETECTED BY TESSERACT>'}")
    print("\n" + "="*50)

if __name__ == "__main__":
    main()