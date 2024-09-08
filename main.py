import torch
from torch.optim import AdamW
from model_loader import load_model
from adapter_tuning import AdvancedAdapterTuning
from local_instructor import LocalInstructor
from memory_optimization import optimize_memory
from code_generation import CodeGenerator
from ui import create_ui
from utils import set_seed
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def generate_text(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def prepare_data(tokenizer, texts, max_length=128):
    encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings

def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load the model
        print("Loading model...")
        model, tokenizer = load_model()
        print("Model loaded successfully")

        # Initialize advanced adapter tuning
        print("Initializing advanced adapter tuning...")
        advanced_adapter_tuning = AdvancedAdapterTuning(model, adapter_dim=64, num_prefix_tokens=20, lora_rank=4)
        print("Advanced adapter tuning initialized")

        # Initialize local instructor
        print("Initializing local instructor...")
        local_instructor = LocalInstructor(model, tokenizer)
        print("Local instructor initialized")

        # Optimize memory utilization
        print("Optimizing memory...")
        optimize_memory(model)
        print("Memory optimization complete")

        # Initialize code generator
        print("Initializing code generator...")
        code_generator = CodeGenerator(model, tokenizer)
        print("Code generator initialized")

        # Load and prepare dataset
        print("Loading and preparing dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = dataset["text"][:1000]  # Use a subset of 1000 examples for demonstration
        train_texts, val_texts = train_test_split(texts, test_size=0.2, random_state=42)
        
        train_data = prepare_data(tokenizer, train_texts)
        val_data = prepare_data(tokenizer, val_texts)
        print("Dataset prepared")

        # Fine-tuning
        print("Starting fine-tuning with advanced techniques...")
        optimizer = AdamW(advanced_adapter_tuning.get_trainable_params(), lr=5e-5)
        advanced_adapter_tuning.train_adapters(train_data, val_data, optimizer, num_epochs=10, batch_size=16, patience=3)
        print("Fine-tuning complete")

        # Compare original model output with fine-tuned model output
        test_prompt = "Python is a great programming language because"
        print("\nComparing model outputs:")
        print("Original model output:")
        print(generate_text(model, tokenizer, test_prompt))

        print("\nFine-tuned model output:")
        advanced_adapter_tuning.enable_adapters()
        print(generate_text(model, tokenizer, test_prompt))
        advanced_adapter_tuning.disable_adapters()

        # Create Gradio interface
        print("Creating Gradio interface...")
        iface = create_ui(model, tokenizer, advanced_adapter_tuning, local_instructor, code_generator)
        print("Gradio interface created")

        # Launch the interface
        print("Launching the interface...")
        iface.launch(server_name="0.0.0.0", server_port=5000)

    except Exception as e:
        print(f"An error occurred during initialization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
