import torch
from transformers import Swin2SRForImageSuperResolution

def get_pretrained_deblur_model():
    print("[*] Loading Pre-trained Swin2SR Model...")
    model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
    
    # 1. Freeze everything first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze the final convolution head AND the final transformer stage.
    # This gives the model enough "brainpower" to learn heavy deblurring
    # without crashing a 4GB laptop GPU.
    for name, param in model.named_parameters():
        if "swin2sr.layers.3" in name or "swin2sr" not in name:
            param.requires_grad = True
            
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] Efficiency Upgrade Applied. Trainable parameters: {trainable_params:,}")
    
    return model