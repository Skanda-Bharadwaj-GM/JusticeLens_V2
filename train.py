import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import LayoutLMv3Processor
import os

# Import your new modules
from src.data.document_dataset import DocumentForgeryDataset
from src.models.layout_lora import get_lora_model

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Training on: {device}")
    
    BATCH_SIZE = 2 # LoRA makes this highly efficient, so batch size 2 or 4 should fit in your VRAM
    EPOCHS = 5
    LR = 5e-5 # LoRA usually requires a slightly higher LR than full fine-tuning
    
    # 1. Initialize Processor and Model
    processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
    model = get_lora_model(num_labels=2).to(device)
    
    # 2. Setup Data
    train_dataset = DocumentForgeryDataset(
        image_dir='data/raw_images',      # <-- Change this line
        annotation_dir='data/annotations', 
        processor=processor
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    # 4. Training Loop
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move everything to GPU
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            # Forward pass. HuggingFace models calculate loss automatically if 'labels' are provided
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}] | Step [{batch_idx}/{len(train_loader)}] | Loss: {loss.item():.4f}")
                
        print(f"[*] Epoch {epoch+1} Average Loss: {total_loss/len(train_loader):.4f}")
        
        # Save the LoRA weights (These will be a tiny file, just a few MBs!)
        model.save_pretrained(f"checkpoints/justice_lens_lora_ep{epoch+1}")
        print(f"[*] Saved LoRA adapter to checkpoints/justice_lens_lora_ep{epoch+1}")

if __name__ == "__main__":
    os.makedirs('checkpoints', exist_ok=True)
    train()