import torch

class ConversationMemory:
    def __init__(self, max_memory_length=5):
        self.memory = []
        self.max_memory_length = max_memory_length

    def add_to_memory(self, user_input, model_output):
        self.memory.append((user_input, model_output))
        if len(self.memory) > self.max_memory_length:
            self.memory.pop(0)

    def get_context(self):
        return " ".join([f"User: {m[0]} Assistant: {m[1]}" for m in self.memory])

    def clear_memory(self):
        self.memory.clear()

class MemoryAugmentedModel:
    def __init__(self, base_model, tokenizer, memory):
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.memory = memory

    def generate_response(self, user_input, max_length=100):
        context = self.memory.get_context()
        full_input = f"{context}\nUser: {user_input}\nAssistant:"
        
        input_ids = self.tokenizer.encode(full_input, return_tensors="pt")
        output = self.base_model.generate(input_ids, max_length=max_length)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the assistant's response
        assistant_response = response.split("Assistant:")[-1].strip()
        
        self.memory.add_to_memory(user_input, assistant_response)
        return assistant_response
