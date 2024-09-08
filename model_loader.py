import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_path="distilgpt2"):
    print(f"Loading model: {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    model.to(device)
    
    print("Model loaded successfully")
    return model, tokenizer
