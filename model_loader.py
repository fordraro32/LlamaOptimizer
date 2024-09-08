import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path, device):
    print(f"Loading model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    
    model.to(device)
    model.eval()
    
    print("Model loaded successfully")
    return model, tokenizer
