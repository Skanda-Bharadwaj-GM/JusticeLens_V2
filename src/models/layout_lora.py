from transformers import LayoutLMv3ForSequenceClassification
from peft import LoraConfig, get_peft_model

def get_lora_model(num_labels=2):
    # 1. Load the heavy pre-trained model
    model = LayoutLMv3ForSequenceClassification.from_pretrained(
        "microsoft/layoutlmv3-base", 
        num_labels=num_labels
    )
    
    # 2. Configure LoRA
    # We target the 'query' and 'value' attention matrices. 
    # This reduces trainable parameters by ~98%.
    config = LoraConfig(
        r=8, # Rank of the update matrices
        lora_alpha=16,
        target_modules=["query", "value"], 
        lora_dropout=0.1,
        bias="none",
        modules_to_save=["classifier"] # We must train the final classification head
    )
    
    # 3. Inject LoRA adapters
    lora_model = get_peft_model(model, config)
    lora_model.print_trainable_parameters() # This will show you exactly how much memory you're saving
    
    return lora_model