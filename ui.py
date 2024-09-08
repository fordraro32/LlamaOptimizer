import gradio as gr

def create_ui(memory_augmented_model, tokenizer, adapter_tuning, local_instructor, code_generator):
    def process_message(message, history):
        # Determine the type of request based on the message content
        if message.lower().startswith("generate code:"):
            response = code_generator.generate_code(message[len("generate code:"):].strip())
        elif message.lower().startswith("explain code:"):
            response = code_generator.explain_code_generation(message[len("explain code:"):].strip())
        else:
            response = memory_augmented_model.generate_response(message)
        
        return response

    def get_conversation_history():
        return "\n".join([f"User: {m[0]}\nAssistant: {m[1]}" for m in memory_augmented_model.memory.memory])

    with gr.Blocks() as interface:
        gr.Markdown("# LLM Optimizer and Fine-Tuning Application")
        
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Enter your message")
        clear = gr.Button("Clear")

        msg.submit(process_message, [msg, chatbot], [chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

        gr.Markdown("## Conversation History")
        history_display = gr.Textbox(label="Conversation History", interactive=False)
        show_history = gr.Button("Show Conversation History")
        show_history.click(get_conversation_history, None, history_display)

    return interface
