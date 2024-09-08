import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
from model_loader import load_model
from adapter_tuning import AdapterTuning
from local_instructor import LocalInstructor
from memory_optimization import optimize_memory
from code_generation import CodeGenerator
from ui import create_ui
from utils import set_seed

# Set random seed for reproducibility
set_seed(42)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model
model_path = "mlabonne/Hermes-3-Llama-3.1-70B-lorablated"
model, tokenizer = load_model(model_path, device)

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
