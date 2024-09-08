import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

def load_model(model_path="/home/ubuntu/LlamaOptimizer/mlabonne_Hermes-3-Llama-3.1-70B-lorablated"):
    print(f"Loading model: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load configuration and modify rope_scaling before converting to config object
    config_dict = AutoConfig.from_pretrained(model_path).to_dict()
    if 'rope_scaling' in config_dict:
        config_dict['rope_scaling'] = {
            "type": "linear",  # Example: set type to "linear" or the appropriate type for your model
            "factor": config_dict['rope_scaling'].get("factor", 8.0)
        }
    config = AutoConfig.from_dict(config_dict)

    # Load model with modified configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )

    model.to(device)

    print("Model loaded successfully")
    return model, tokenizer