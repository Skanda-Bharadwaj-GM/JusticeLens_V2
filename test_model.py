import torch
import numpy as np
from PIL import Image
from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
HF_USERNAME = "skandab17" # Replace with your actual Hugging Face username
REPO_ID = f"{HF_USERNAME}/justice-lens-weights"
WEIGHT_FILE = "lens_cloud_latest_checkpoint.pth" # Change this if you want to test epoch 1, 2, etc.

INPUT_IMAGE = "0066645_blur.png"  # The blurry image you want to test
OUTPUT_IMAGE = "restored_ocr_clear.jpg" # What the AI will save the fixed image as

def main():
    print(f"[*] Downloading {WEIGHT_FILE} from Hugging Face...")
    try:
        # This automatically pulls your trained weights from the cloud!
        weights_path = hf_hub_download(repo_id=REPO_ID, filename=WEIGHT_FILE)
    except Exception as e:
        print(f"[ERROR] Could not find the file on Hugging Face. Is the training finished? \n{e}")
        return

    print("[*] Loading the base Swin2SR Brain...")
    processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
    model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")

    print("[*] Injecting your custom Justice Lens knowledge...")
    # Load your custom weights into the model
    custom_weights = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(custom_weights)
    model.eval() # Set to evaluation mode

    print(f"[*] Analyzing {INPUT_IMAGE}...")
    image = Image.open(INPUT_IMAGE).convert("RGB")
    inputs = processor(image, return_tensors="pt")

    # Run the image through the model without tracking gradients (saves memory)
    with torch.no_grad():
        outputs = model(**inputs)

    print("[*] Reconstructing the clear image...")
    # Convert the mathematical tensor back into a normal image format
    output_tensor = outputs.reconstruction.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output_tensor = np.transpose(output_tensor, (1, 2, 0))
    output_image = (output_tensor * 255.0).round().astype(np.uint8)
    
    restored_img = Image.fromarray(output_image)
    restored_img.save(OUTPUT_IMAGE)
    
    print(f"[SUCCESS] The deblurred image has been saved as: {OUTPUT_IMAGE}")

if __name__ == "__main__":
    main()