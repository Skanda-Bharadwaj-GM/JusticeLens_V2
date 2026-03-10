import cv2
import pytesseract
import jiwer
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

# --- CONFIGURATION ---
RESTORED_IMG_PATH = "restored_ocr_clear.jpg"  # The image your AI output
SHARP_IMG_PATH = "0066645_orig.png"    # The original, perfectly clear image
# ---------------------
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def evaluate_image():
    print("[*] Loading images for evaluation...")
    # Load images in grayscale (best for evaluating text structure)
    restored = cv2.imread(RESTORED_IMG_PATH, cv2.IMREAD_GRAYSCALE)
    sharp = cv2.imread(SHARP_IMG_PATH, cv2.IMREAD_GRAYSCALE)

    # Ensure images are the exact same size for pixel comparison
    if restored.shape != sharp.shape:
        restored = cv2.resize(restored, (sharp.shape[1], sharp.shape[0]))

    print("\n" + "="*40)
    print(" IMAGE QUALITY METRICS")
    print("="*40)

    # 1. Calculate PSNR
    psnr_value = compare_psnr(sharp, restored)
    print(f"PSNR (Pixel Accuracy): {psnr_value:.2f} dB")

    # 2. Calculate SSIM
    ssim_value = compare_ssim(sharp, restored)
    print(f"SSIM (Structural Accuracy): {ssim_value:.4f}")

    print("\n" + "="*40)
    print(" OCR TEXT RECOVERY METRICS")
    print("="*40)

    # 3. Extract Text via Tesseract
    sharp_text = pytesseract.image_to_string(sharp).strip()
    restored_text = pytesseract.image_to_string(restored).strip()

    # Calculate Character Error Rate (CER)
    if sharp_text == "":
        print("[!] Could not detect text in the sharp ground truth image.")
    else:
        # jiwer.cer calculates the character-level edit distance
        error_rate = jiwer.cer(sharp_text, restored_text)
        accuracy = max(0, 1.0 - error_rate) * 100
        
        print(f"Ground Truth Text Length: {len(sharp_text)} characters")
        print(f"Character Error Rate (CER): {error_rate:.4f}")
        print(f"Estimated Text Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    evaluate_image()