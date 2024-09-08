import torch
import torch.nn as nn
import torch.optim as optim

class LocalInstructor:
    def __init__(self, model, tokenizer, learning_rate=1e-5):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)

    def prepare_data(self, instructions):
        inputs = self.tokenizer(instructions, return_tensors="pt", padding=True, truncation=True)
        return inputs

    def train(self, instructions, num_epochs=3):
        self.model.train()
        inputs = self.prepare_data(instructions)

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()
            outputs = self.model(**inputs)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        self.model.eval()

    def generate_response(self, prompt, max_length=100):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
