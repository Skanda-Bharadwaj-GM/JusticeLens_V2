import os
from huggingface_hub import hf_hub_download

REPO_ID = "skandab17/justice-lens-weights"
FILENAME = "lens_cloud_latest_checkpoint.pth"
LOCAL_DIR = "./models"

def fetch_weights():
    print(f"[*] Checking for Justice Lens weights in {LOCAL_DIR}...")
    
    # Create the folder if it doesn't exist
    os.makedirs(LOCAL_DIR, exist_ok=True)
    local_path = os.path.join(LOCAL_DIR, FILENAME)

    if not os.path.exists(local_path):
        print(f"[*] Weights not found locally. Downloading from Hugging Face ({REPO_ID})...")
        # Downloads directly to your local folder
        hf_hub_download(
            repo_id=REPO_ID, 
            filename=FILENAME, 
            local_dir=LOCAL_DIR
        )
        print("[*] Download complete! Ready for inference.")
    else:
        print("[*] Weights already exist locally. Skipping download.")

if __name__ == "__main__":
    fetch_weights()