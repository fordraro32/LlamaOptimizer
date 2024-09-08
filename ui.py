import gradio as gr

def create_ui(model, tokenizer, adapter_tuning, local_instructor, code_generator):
    conversation_history = []

    def process_message(message, history):
        conversation_history.append(("user", message))
        
        # Determine the type of request based on the message content
        if message.lower().startswith("generate code:"):
            response = code_generator.generate_code(message[len("generate code:"):].strip())
        elif message.lower().startswith("explain code:"):
            response = code_generator.explain_code_generation(message[len("explain code:"):].strip())
        else:
            response = generate_text(message)
        
        conversation_history.append(("assistant", response))
        return response

    def generate_text(prompt, max_length=100):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(input_ids, max_length=max_length)
        return tokenizer.decode(output[0], skip_special_tokens=True)

    def get_conversation_history():
        return "\n".join([f"{'User' if role == 'user' else 'Assistant'}: {msg}" for role, msg in conversation_history])

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
