import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from .advanced_techniques import apply_mixout, R3FLoss  # Changed to relative import

...


class AdapterLayer(nn.Module):

    def __init__(self, input_dim, adapter_dim):
        super().__init__()
        self.down_project = nn.Linear(input_dim, adapter_dim)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_dim, input_dim)

    def forward(self, x):
        return self.up_project(self.activation(self.down_project(x))) + x


class PrefixTuning(nn.Module):

    def __init__(self, config, num_prefix_tokens=20):
        super().__init__()
        self.num_prefix_tokens = num_prefix_tokens
        self.prefix_embedding = nn.Embedding(num_prefix_tokens,
                                             config.hidden_size)
        self.prefix_projection = nn.Linear(config.hidden_size,
                                           config.hidden_size)

    def forward(self, hidden_states):
        prefix = self.prefix_embedding(
            torch.arange(self.num_prefix_tokens, device=hidden_states.device))
        prefix = self.prefix_projection(prefix)
        return torch.cat([prefix, hidden_states], dim=1)


class LoRA(nn.Module):

    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = 0.01

    def forward(self, x):
        return self.scaling * (x @ self.lora_A.T @ self.lora_B.T)


class AdvancedAdapterTuning:

    def __init__(self,
                 model,
                 adapter_dim=64,
                 num_prefix_tokens=20,
                 lora_rank=4,
                 mixout_prob=0.1,
                 device='cuda'):
        self.model = model.to(
            device)  # Ensure the model is on the correct device
        self.adapter_dim = adapter_dim
        self.num_prefix_tokens = num_prefix_tokens
        self.lora_rank = lora_rank
        self.mixout_prob = mixout_prob
        self.device = device
        self.adapters = nn.ModuleDict()
        self.prefix_tuning = None
        self.loras = nn.ModuleDict()
        self.add_adapters()
        self.apply_mixout()

    def add_adapters(self):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                adapter = AdapterLayer(module.out_features,
                                       self.adapter_dim).to(self.device)
                self.adapters[name] = adapter

                lora = LoRA(module.in_features, module.out_features,
                            self.lora_rank).to(self.device)
                self.loras[name] = lora

        # Add prefix tuning
        self.prefix_tuning = PrefixTuning(
            self.model.config, self.num_prefix_tokens).to(self.device)

    def apply_mixout(self):
        self.model = apply_mixout(self.model, p=self.mixout_prob)

    def adapter_forward(self, name):

        def hook(module, input, output):
            # Ensure input and output tensors are transferred to the same device
            output = self.adapters[name](output.to(self.device)).to(
                self.device)
            input_lora = input[0].to(
                self.device
            )  # Ensure the input for LoRA is on the correct device
            output = output + self.loras[name](input_lora)
            return output

        return hook

    def enable_adapters(self):
        for name, module in self.model.named_modules():
            if name in self.adapters:
                module.register_forward_hook(self.adapter_forward(name))

        # Enable prefix tuning
        first_layer = next(self.model.children())
        if isinstance(first_layer, nn.Embedding):
            original_forward = first_layer.forward

            def new_forward(input_ids):
                embeddings = original_forward(input_ids)
                return self.prefix_tuning(embeddings)

            first_layer.forward = new_forward

    def disable_adapters(self):
        for module in self.model.modules():
            module._forward_hooks.clear()

        # Disable prefix tuning
        first_layer = next(self.model.children())
        if isinstance(first_layer, nn.Embedding):
            first_layer.forward = first_layer.__class__.forward.__get__(
                first_layer, first_layer.__class__)

    def gradual_unfreeze(self, epoch, total_epochs):
        n_layers = len(list(self.model.named_children()))
        n_unfrozen = int((epoch + 1) / total_epochs * n_layers)
        for i, (name, param) in enumerate(
                reversed(list(self.model.named_parameters()))):
            param.requires_grad = i < n_unfrozen

    def get_layer_optimizer(self, lr):
        layer_parameters = []
        for name, param in self.model.named_parameters():
            layer = int(name.split('.')[2]) if 'layer' in name else 0
            layer_parameters.append({
                'params': param,
                'lr': lr * (0.95**layer)
            })
        return torch.optim.AdamW(layer_parameters)

    def train_adapters(self,
                       train_data,
                       val_data,
                       num_epochs,
                       batch_size=8,
                       patience=3,
                       lr=5e-5):
        self.enable_adapters()
        self.model.train()
        loss_fn = nn.CrossEntropyLoss()
        r3f_loss = R3FLoss(self.model)

        train_loader = DataLoader(
            TensorDataset(*[v.to(self.device) for v in train_data.values()]),
            batch_size=batch_size,
            shuffle=True)
        val_loader = DataLoader(
            TensorDataset(*[v.to(self.device) for v in val_data.values()]),
            batch_size=batch_size)

        best_val_loss = float('inf')
        epochs_without_improvement = 0

        optimizer = self.get_layer_optimizer(lr)

        for epoch in range(num_epochs):
            self.gradual_unfreeze(epoch, num_epochs)

            total_train_loss = 0
            self.model.train()

            for batch in tqdm(train_loader,
                              desc=f"Epoch {epoch + 1}/{num_epochs}"):
                optimizer.zero_grad()
                inputs = {
                    k: v.to(self.device)
                    for k, v in zip(train_data.keys(), batch)
                }
                outputs = self.model(**inputs)
                logits = outputs.logits

                # Shift labels for causal language modeling
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = inputs['input_ids'][..., 1:].contiguous()

                ce_loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)),
                                  shift_labels.view(-1))
                r3f = r3f_loss(inputs)
                loss = ce_loss + 0.1 * r3f  # Add R3F loss with a weight of 0.1

                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation
            self.model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    inputs = {
                        k: v.to(self.device)
                        for k, v in zip(val_data.keys(), batch)
                    }
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = inputs['input_ids'][..., 1:].contiguous()

                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
            )

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        self.model.eval()

    def get_trainable_params(self):
        return list(self.adapters.parameters()) + list(
            self.prefix_tuning.parameters()) + list(self.loras.parameters())
