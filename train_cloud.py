import os
from huggingface_hub import HfApi
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
    BATCH_SIZE = 4           
    ACCUMULATION_STEPS = 2   
    EPOCHS = 5
    LR = 1e-4 
    
    processor = Swin2SRImageProcessor()
    model = get_pretrained_deblur_model().to(device)
    
    # Using 512 for maximum resolution context
    dataset = DocumentDeblurDataset(
        blur_dir='data/blur_docs', 
        sharp_dir='data/sharp_docs', 
        processor=processor,
        patch_size=512 
    )
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.L1Loss()
    
    os.makedirs('/app/checkpoints', exist_ok=True) # Cloud absolute path for outputs
    
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
            
        print(f"[*] Epoch {epoch+1} Average Loss: {total_loss/len(loader):.4f}")
        # Save locally in the container
        save_path = f"/app/checkpoints/lens_cloud_ep{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        
        # Upload directly to Hugging Face
        print("[*] Uploading checkpoint to Hugging Face...")
        try:
            api = HfApi()
            api.upload_file(
                path_or_fileobj=save_path,
                path_in_repo=f"lens_cloud_ep{epoch+1}.pth",
                repo_id="skandab17/justice-lens-weights", # <-- Put your HF username here
                token=os.environ.get("HF_TOKEN") # We will securely pass this in Salad
            )
            print("[*] Upload successful!")
        except Exception as e:
            print(f"[!] Upload failed: {e}")

if __name__ == "__main__":
    setup_cloud_data()
    train_cloud()
    print("[*] Cloud training complete. Shutting down node.")