import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import Swin2SRImageProcessor
import os
from tqdm import tqdm

from src.models.deblur_model import get_pretrained_deblur_model
from src.data.deblur_dataset import DocumentDeblurDataset

def train_stage1():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training Stage 1 (Deblurring) on: {device}")
    
    BATCH_SIZE = 1           
    ACCUMULATION_STEPS = 4   
    EPOCHS = 10
    LR = 1e-4 
    
    processor = Swin2SRImageProcessor()
    model = get_pretrained_deblur_model().to(device)
    
    os.makedirs('data/blur_docs', exist_ok=True)
    os.makedirs('data/sharp_docs', exist_ok=True)
    
    # --- THE MAX RESOLUTION UPGRADE ---
    # We are pushing for 512x512 patches to give the model maximum context.
    # If your GPU crashes with a CUDA Out Of Memory error, change patch_size to 384.
    dataset = DocumentDeblurDataset(
        blur_dir='data/blur_docs', 
        sharp_dir='data/sharp_docs', 
        processor=processor,
        patch_size=512 
    )
    
    if len(dataset) == 0:
        print("[!] No images found in data/blur_docs. Please add training data.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.L1Loss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        optimizer.zero_grad() 
        
        loop = tqdm(loader, leave=True)
        
        for idx, batch in enumerate(loop):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            # Standard 32-bit Forward Pass (Stable)
            outputs = model(pixel_values)
            reconstruction = outputs.reconstruction
            
            if reconstruction.shape != labels.shape:
                reconstruction = torch.nn.functional.interpolate(reconstruction, size=labels.shape[-2:])
                
            loss = criterion(reconstruction, labels)
            loss = loss / ACCUMULATION_STEPS
            
            # Backward Pass
            loss.backward()
            
            if ((idx + 1) % ACCUMULATION_STEPS == 0) or (idx + 1 == len(loader)):
                # Safety valve to prevent 'nan' loss
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
            
            actual_loss = loss.item() * ACCUMULATION_STEPS
            total_loss += actual_loss
            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=actual_loss)
            
        print(f"[*] Epoch {epoch+1} Average Loss: {total_loss/len(loader):.4f}")
        
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/lens_audit_deblur_ep{epoch+1}.pth")
        
if __name__ == "__main__":
    train_stage1()