import torch

class CodeGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_code(self, prompt, max_length=500, temperature=0.7, top_p=0.95):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=1,
            )

        generated_code = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_code

    def explain_code_generation(self, code, max_length=300):
        prompt = f"Explain the following code:\n\n{code}\n\nExplanation:"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                num_return_sequences=1,
            )

        explanation = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return explanation
