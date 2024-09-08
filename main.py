import torch
from model_loader import load_model
from adapter_tuning import AdapterTuning
from local_instructor import LocalInstructor
from memory_optimization import optimize_memory
from code_generation import CodeGenerator
from ui import create_ui
from utils import set_seed

def main():
    # Set random seed for reproducibility
    set_seed(42)

    # Check for CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        # Load the model
        model, tokenizer = load_model()

        # Initialize adapter tuning
        adapter_tuning = AdapterTuning(model)

        # Initialize local instructor
        local_instructor = LocalInstructor(model, tokenizer)

        # Optimize memory utilization
        optimize_memory(model)

        # Initialize code generator
        code_generator = CodeGenerator(model, tokenizer)

        # Create Gradio interface
        iface = create_ui(model, tokenizer, adapter_tuning, local_instructor, code_generator)

        # Launch the interface
        iface.launch(server_name="0.0.0.0", server_port=5000)

    except Exception as e:
        print(f"An error occurred during initialization: {str(e)}")
        raise

if __name__ == "__main__":
    main()
