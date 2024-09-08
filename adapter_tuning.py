import torch
import torch.nn as nn

class AdapterLayer(nn.Module):
    def __init__(self, input_dim, adapter_dim):
        super().__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, input_dim)

    def forward(self, x):
        return self.up_project(self.activation(self.down_project(x))) + x

class AdapterTuning:
    def __init__(self, model, adapter_dim=64):
        self.model = model
        self.adapter_dim = adapter_dim
        self.adapters = nn.ModuleDict()
        self.add_adapters()

    def add_adapters(self):
        for name, module in self.model.named_children():
            if isinstance(module, nn.Linear):
                adapter = AdapterLayer(module.out_features, self.adapter_dim)
                self.adapters[name] = adapter

    def adapter_forward(self, name):
        def hook(module, input, output):
            return self.adapters[name](output)
        return hook

    def enable_adapters(self):
        for name, module in self.model.named_children():
            if name in self.adapters:
                module.register_forward_hook(self.adapter_forward(name))

    def disable_adapters(self):
        for module in self.model.modules():
            module._forward_hooks.clear()

    def train_adapters(self, train_data, optimizer, num_epochs):
        self.enable_adapters()
        self.model.train()

        for epoch in range(num_epochs):
            for batch in train_data:
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        self.model.eval()
