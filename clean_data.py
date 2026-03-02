import os
from PIL import Image

def clean_corrupted_images(directory):
    removed = 0
    files = [f for f in os.listdir(directory) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"[*] Scanning {len(files)} images for corruption...")
    
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        try:
            # Try to fully load the image data
            img = Image.open(file_path)
            img.verify() # Checks for broken files
        except Exception as e:
            print(f"[!] Found corrupted file: {file_name}. Deleting...")
            os.remove(file_path)
            removed += 1
            
    print(f"[*] Done! Removed {removed} corrupted images.")

if __name__ == "__main__":
    clean_corrupted_images('data/sharp_docs')