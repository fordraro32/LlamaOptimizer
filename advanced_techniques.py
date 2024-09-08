import torch
import torch.nn as nn
import torch.nn.functional as F

class Mixout(nn.Module):
    def __init__(self, module, p=0.2):
        super().__init__()
        self.module = module
        self.p = p
        self.pretrained_weight = module.weight.data.clone()

    def forward(self, input):
        if self.training:
            mask = torch.bernoulli(torch.full_like(self.module.weight, self.p))
            self.module.weight.data = mask * self.pretrained_weight + (1 - mask) * self.module.weight.data
        return self.module(input)

def apply_mixout(model, p=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            setattr(model, name, Mixout(module, p))
    return model

def r3f_loss(model, inputs, epsilon=1e-3):
    original_embeds = model.get_input_embeddings()(inputs['input_ids'])
    perturbed_embeds = original_embeds + epsilon * torch.randn_like(original_embeds)
    
    original_outputs = model(inputs_embeds=original_embeds, attention_mask=inputs['attention_mask'])
    perturbed_outputs = model(inputs_embeds=perturbed_embeds, attention_mask=inputs['attention_mask'])
    
    kl_loss = F.kl_div(
        F.log_softmax(perturbed_outputs.logits, dim=-1),
        F.softmax(original_outputs.logits, dim=-1),
        reduction='batchmean'
    )
    return kl_loss

class R3FLoss(nn.Module):
    def __init__(self, model, epsilon=1e-3):
        super().__init__()
        self.model = model
        self.epsilon = epsilon

    def forward(self, inputs):
        return r3f_loss(self.model, inputs, self.epsilon)
