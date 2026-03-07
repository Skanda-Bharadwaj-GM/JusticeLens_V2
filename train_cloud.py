import os
from huggingface_hub import HfApi, hf_hub_download
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Swin2SRImageProcessor
from tqdm import tqdm
import gdown

from src.models.deblur_model import get_pretrained_deblur_model
from src.data.deblur_dataset import DocumentDeblurDataset

# --- CLOUD CONFIGURATION ---
GDRIVE_URL = "https://drive.google.com/file/d/1Q7GEYmjTptog4hZD0UXFnmc5DJsPoMI9/view?usp=drive_link" 
ZIP_NAME = "justice_lens_data.zip"
EXTRACT_DIR = "." # Extracts directly into the current directory
REPO_ID = "skandab17/justice-lens-weights" # Centralized your repo ID here

def setup_cloud_data():
    print("[*] Downloading dataset from Google Drive...")
    # gdown automatically handles the fuzzy Google Drive share links
    gdown.download(GDRIVE_URL, ZIP_NAME, quiet=False, fuzzy=True)
    
    print("[*] Extracting dataset...")
    with zipfile.ZipFile(ZIP_NAME, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)
    
    print("[*] Dataset ready. Cleaning up zip file...")
    os.remove(ZIP_NAME)

def train_cloud():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing Cloud Training on: {device}")
    
    # We can push the batch size higher on a cloud GPU (e.g., 24GB VRAM)
    BATCH_SIZE = 1           
    ACCUMULATION_STEPS = 8  
    EPOCHS = 25 # UPGRADED TO 25 EPOCHS
    LR = 1e-4 
    
    processor = Swin2SRImageProcessor()
    model = get_pretrained_deblur_model().to(device)
    
    # --- THE BRAIN SURGERY (UNFREEZING DEEP LAYERS) ---
    print("[*] Initiating Deep Layer Unfreezing...")
    # 1. Freeze everything first to establish a baseline
    for param in model.parameters():
        param.requires_grad = False

    # 2. Wake up the final transformer block and upsampling layers
    for name, param in model.named_parameters():
        if "layers.3" in name or "conv_after_body" in name or "upsample" in name:
            param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Upgraded Trainable Parameters: {trainable_params:,}")
    # ---------------------------------------------------

    # --- ANTI-PREEMPTION RESUME ---
    print("[*] Checking Hugging Face for interrupted checkpoints...")
    hf_token = os.environ.get("HF_TOKEN")
    try:
        # If a node died mid-training, the new node will download the last save state here
        checkpoint_path = hf_hub_download(repo_id=REPO_ID, filename="lens_cloud_latest_checkpoint.pth", token=hf_token)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print("[*] SUCCESS: Checkpoint found! Resuming from previous state.")
    except Exception:
        print("[*] No checkpoint found on Hugging Face. Starting fresh.")
    # ------------------------------

    # Using 512 for maximum resolution context
    dataset = DocumentDeblurDataset(
        blur_dir='data/blur_docs', 
        sharp_dir='data/sharp_docs', 
        processor=processor,
        patch_size=256 
    )
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Optimizer must be initialized AFTER unfreezing layers so it tracks the newly active weights
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.L1Loss()
    
    os.makedirs('/app/checkpoints', exist_ok=True) # Cloud absolute path for outputs
    
    api = HfApi() # Initialize API once for the whole loop

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad() 
        
        loop = tqdm(loader, leave=True)
        for idx, batch in enumerate(loop):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(pixel_values)
            reconstruction = outputs.reconstruction
            
            if reconstruction.shape != labels.shape:
                reconstruction = torch.nn.functional.interpolate(reconstruction, size=labels.shape[-2:])
                
            loss = criterion(reconstruction, labels)
            loss = loss / ACCUMULATION_STEPS
            loss.backward()
            
            if ((idx + 1) % ACCUMULATION_STEPS == 0) or (idx + 1 == len(loader)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            actual_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += actual_loss
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=actual_loss)

            # --- ANTI-PREEMPTION CLOUD SAVE (MID-EPOCH) ---
            # Every 500 batches, securely back up the model to Hugging Face
            if idx % 500 == 0 and idx > 0:
                print(f"\n[*] Anti-Preemption Triggered: Securing checkpoint at batch {idx}...")
                checkpoint_filename = "/app/checkpoints/lens_cloud_latest_checkpoint.pth"
                torch.save(model.state_dict(), checkpoint_filename)
                
                try:
                    api.upload_file(
                        path_or_fileobj=checkpoint_filename,
                        path_in_repo="lens_cloud_latest_checkpoint.pth",
                        repo_id=REPO_ID,
                        token=hf_token,
                        repo_type="model"
                    )
                    print("[*] Mid-epoch checkpoint successfully secured in the cloud.")
                except Exception as e:
                    print(f"[!] Failed to upload checkpoint: {e}")
            # ----------------------------------------------
            
        print(f"[*] Epoch {epoch+1} Average Loss: {total_loss/len(loader):.4f}")
        # Save locally in the container
        save_path = f"/app/checkpoints/lens_cloud_ep{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        
        # Upload directly to Hugging Face
        print("[*] Uploading checkpoint to Hugging Face...")
        try:
            api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=f"lens_cloud_ep{epoch+1}.pth",
                repo_id=REPO_ID, 
                token=hf_token 
            )
            print("[*] Upload successful!")
        except Exception as e:
            print(f"[!] Upload failed: {e}")

if __name__ == "__main__":
    setup_cloud_data()
    train_cloud()
    print("[*] Cloud training complete. Shutting down node.")