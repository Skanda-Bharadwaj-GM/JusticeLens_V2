import os
import shutil
import cv2
import numpy as np
import random

def apply_motion_blur(image, size=15):
    kernel = np.zeros((size, size))
    angle = random.choice([0, 45, 90, 135])
    if angle == 0: kernel[int((size-1)/2), :] = np.ones(size)
    elif angle == 90: kernel[:, int((size-1)/2)] = np.ones(size)
    else:
        for i in range(size):
            if angle == 45: kernel[i, i] = 1
            elif angle == 135: kernel[i, size - i - 1] = 1
    return cv2.filter2D(image, -1, kernel / size)

def apply_defocus_blur(image, kernel_size=5):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

print("\n--- MASTER DATA PREP ---")
sharp_dir = 'data/sharp_docs'
blur_dir = 'data/blur_docs'

# 1. Nuke the out-of-sync folder
if os.path.exists(blur_dir):
    print(f"[*] Deleting out-of-sync {blur_dir} folder...")
    shutil.rmtree(blur_dir)
os.makedirs(blur_dir)

# 2. Count the sharp files
sharp_files = [f for f in os.listdir(sharp_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
print(f"[*] Found {len(sharp_files)} perfectly clean receipts in {sharp_dir}.")

if len(sharp_files) == 0:
    print("[!] ERROR: No images found! Did they get accidentally moved?")
    print("[!] Please put your SROIE receipts back into data/sharp_docs and run this again.")
    exit()

# 3. Generate exactly matching twins
print("[*] Generating perfectly synced blurry twins...")
for file_name in sharp_files:
    sharp_path = os.path.join(sharp_dir, file_name)
    blur_path = os.path.join(blur_dir, file_name)
    
    img = cv2.imread(sharp_path)
    if img is None: continue
    
    dt = random.choice(['motion', 'defocus', 'both'])
    if dt in ['motion', 'both']: img = apply_motion_blur(img, size=random.choice([9, 11, 15]))
    if dt in ['defocus', 'both']: img = apply_defocus_blur(img, kernel_size=random.choice([3, 5, 7]))
        
    noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    cv2.imwrite(blur_path, img)

print(f"[*] Success! {len(sharp_files)} pairs are now 100% mathematically synced.")
print("[*] You may now run: python train_deblur.py")