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
import logging
import sys
from conversation_memory import ConversationMemory, MemoryAugmentedModel
from plugin_system import PluginManager, GradientClippingPlugin

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def generate_text(model, tokenizer, prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    output = model.generate(input_ids, max_length=max_length)
    return tokenizer.decode(output[0], skip_special_tokens=True)


def prepare_data(tokenizer, texts, max_length=128):
    encodings = tokenizer(texts,
                          truncation=True,
                          padding="max_length",
                          max_length=max_length,
                          return_tensors="pt")
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings


def main():
    try:
        # Set random seed for reproducibility
        set_seed(42)

        # Check for CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {device}")

        # Load the model
        logging.info("Loading model...")
        model, tokenizer = load_model()
        model.to(device)  # Move the model to the appropriate device
        logging.info("Model loaded successfully")

        # Initialize conversation memory
        logging.info("Initializing conversation memory...")
        memory = ConversationMemory(max_memory_length=5)
        memory_augmented_model = MemoryAugmentedModel(model, tokenizer, memory)
        logging.info("Conversation memory initialized")

        # Initialize plugin manager and register plugins
        logging.info("Initializing plugin manager...")
        plugin_manager = PluginManager()
        plugin_manager.register_plugin("gradient_clipping",
                                       GradientClippingPlugin(max_norm=1.0))
        logging.info("Plugin manager initialized")

        # Initialize advanced adapter tuning with new techniques
        logging.info("Initializing advanced adapter tuning...")
        advanced_adapter_tuning = AdvancedAdapterTuning(model,
                                                        adapter_dim=64,
                                                        num_prefix_tokens=20,
                                                        lora_rank=4,
                                                        mixout_prob=0.1,
                                                        device=device)
        logging.info("Advanced adapter tuning initialized")

        # Initialize local instructor
        logging.info("Initializing local instructor...")
        local_instructor = LocalInstructor(model, tokenizer)
        logging.info("Local instructor initialized")

        # Optimize memory utilization
        logging.info("Optimizing memory...")
        optimize_memory(model)
        logging.info("Memory optimization complete")

        # Initialize code generator
        logging.info("Initializing code generator...")
        code_generator = CodeGenerator(model, tokenizer)
        logging.info("Code generator initialized")

        # Load and prepare dataset
        logging.info("Loading and preparing dataset...")
        try:
            dataset = load_dataset("wikitext",
                                   "wikitext-2-raw-v1",
                                   split="train")
            texts = dataset[
                "text"][:
                        1000]  # Use a subset of 1000 examples for demonstration
            train_texts, val_texts = train_test_split(texts,
                                                      test_size=0.2,
                                                      random_state=42)

            train_data = prepare_data(tokenizer, train_texts)
            val_data = prepare_data(tokenizer, val_texts)
            logging.info("Dataset prepared successfully")
        except Exception as e:
            logging.error(f"Error in dataset preparation: {str(e)}")
            raise

        # Fine-tuning with advanced techniques
        logging.info("Starting fine-tuning with advanced techniques...")
        try:
            # Apply all optimization plugins
            model = plugin_manager.apply_all_plugins(model)

            advanced_adapter_tuning.train_adapters(train_data,
                                                   val_data,
                                                   num_epochs=10,
                                                   batch_size=16,
                                                   patience=3,
                                                   lr=5e-5)
            logging.info("Fine-tuning complete")
        except Exception as e:
            logging.error(f"Error during fine-tuning: {str(e)}")
            raise

        # Compare original model output with fine-tuned model output
        test_prompt = "Python is a great programming language because"
        logging.info("\nComparing model outputs:")
        logging.info("Original model output:")
        logging.info(generate_text(model, tokenizer, test_prompt))

        logging.info("\nFine-tuned model output:")
        advanced_adapter_tuning.enable_adapters()
        logging.info(generate_text(model, tokenizer, test_prompt))
        advanced_adapter_tuning.disable_adapters()

        # Create Gradio interface
        logging.info("Creating Gradio interface...")
        iface = create_ui(memory_augmented_model, tokenizer,
                          advanced_adapter_tuning, local_instructor,
                          code_generator)
        logging.info("Gradio interface created")

        # Launch the interface
        logging.info("Launching the interface...")
        iface.launch(server_name="0.0.0.0", server_port=5000)

    except Exception as e:
        logging.error(f"An error occurred during initialization: {str(e)}")
        raise


if __name__ == "__main__":
    main()
