import difflib
from PIL import Image
import pytesseract

# --- CONFIGURATION ---
# 1. The perfect, clear version of the document (Ground Truth)
ORIGINAL_IMAGE = "0000079_orig.png" 

# 2. The image you are testing (The blurry one, or the AI-restored one)
TEST_IMAGE = "restored_ocr_clear8.jpg"                         

# Windows Tesseract Path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# ---------------------

def extract_text(image_path):
    """Extracts text from an image using Tesseract OCR."""
    try:
        img = Image.open(image_path)
        # Tesseract usually performs best on standard RGB or Grayscale images
        text = pytesseract.image_to_string(img).strip()
        return text
    except FileNotFoundError:
        print(f"[ERROR] Could not find the file: {image_path}")
        return ""
    except Exception as e:
        print(f"[ERROR] An issue occurred reading {image_path}:\n{e}")
        return ""

def calculate_accuracy(expected_text, extracted_text):
    """Calculates the similarity percentage between two text strings."""
    if not expected_text or not extracted_text:
        return 0.0
    # Lowercase both strings so capitalization errors don't tank the score
    return difflib.SequenceMatcher(None, expected_text.lower(), extracted_text.lower()).ratio() * 100

def main():
    print("[*] Starting OCR Image-to-Image Comparison...\n")

    # 1. Extract Ground Truth from the original clear image
    print(f"[*] Reading ORIGINAL image: {ORIGINAL_IMAGE}")
    ground_truth_text = extract_text(ORIGINAL_IMAGE)

    if not ground_truth_text:
        print("[!] WARNING: No text found in the original image. The comparison will fail.")
        return

    # 2. Extract Text from the test image
    print(f"[*] Reading TEST image: {TEST_IMAGE}")
    test_text = extract_text(TEST_IMAGE)

    # 3. Calculate the match percentage
    accuracy = calculate_accuracy(ground_truth_text, test_text)

    # 4. Print the final report
    print("\n" + "="*60)
    print(" IMAGE-TO-IMAGE OCR ACCURACY REPORT")
    print("="*60)
    
    print("\n[GROUND TRUTH: Original Image Text]")
    print(f"{ground_truth_text}")
    
    print("\n" + "-"*60)
    
    print("\n[TEST OUTPUT: Target Image Text]")
    print(f"{test_text if test_text else '<NO TEXT DETECTED>'}")
    
    print("\n" + "="*60)
    print(f" FINAL MATCH ACCURACY: {accuracy:.2f}%")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()