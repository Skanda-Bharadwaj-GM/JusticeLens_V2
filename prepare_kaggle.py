import os
import random
import shutil
from tqdm import tqdm

# --- CONFIGURATION (Absolute Paths) ---
KAGGLE_SHARP_DIR = r"C:\Users\Skanda Bharadwaj G M\Downloads\archive (8)\BMVC_image_data\orig"
KAGGLE_BLUR_DIR = r"C:\Users\Skanda Bharadwaj G M\Downloads\archive (8)\BMVC_image_data\blur"

DEST_SHARP_DIR = r"C:\Users\Skanda Bharadwaj G M\Downloads\Justice_LensV2\data\sharp_docs"
DEST_BLUR_DIR = r"C:\Users\Skanda Bharadwaj G M\Downloads\Justice_LensV2\data\blur_docs"

NUM_IMAGES_TO_EXTRACT = 10000

def prepare_data():
    print("[*] Preparing Cloud Training Payload...")
    
    # 1. Clear out old data
    for folder in [DEST_SHARP_DIR, DEST_BLUR_DIR]:
        os.makedirs(folder, exist_ok=True)
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    print("[*] Cleared local data folders.")

    # 2. Get list of all original images
    all_orig_files = [f for f in os.listdir(KAGGLE_SHARP_DIR) if f.endswith(('.png', '.jpg'))]
    print(f"[*] Found {len(all_orig_files)} total images in Kaggle dataset.")
    
    # 3. Randomly select 10,000 files
    selected_files = random.sample(all_orig_files, min(NUM_IMAGES_TO_EXTRACT, len(all_orig_files)))
    
    # 4. Copy and Rename the pairs
    print(f"[*] Extracting and renaming {len(selected_files)} image pairs...")
    copied_count = 0
    
    for orig_file in tqdm(selected_files):
        # Translate the filename: "0000000_orig.png" -> "0000000_blur.png"
        blur_file = orig_file.replace("_orig", "_blur")
        
        src_sharp = os.path.join(KAGGLE_SHARP_DIR, orig_file)
        src_blur = os.path.join(KAGGLE_BLUR_DIR, blur_file)
        
        # Strip the suffixes so PyTorch dataloader can match them later
        # "0000000_orig.png" becomes "0000000.png"
        final_name = orig_file.replace("_orig", "")
        
        dst_sharp = os.path.join(DEST_SHARP_DIR, final_name)
        dst_blur = os.path.join(DEST_BLUR_DIR, final_name)
        
        # Safety check: ensure both physically exist before copying
        if os.path.exists(src_sharp) and os.path.exists(src_blur):
            shutil.copy2(src_sharp, dst_sharp)
            shutil.copy2(src_blur, dst_blur)
            copied_count += 1
            
    print(f"\n[*] Data preparation complete! Successfully copied {copied_count} pairs.")

if __name__ == "__main__":
    prepare_data()