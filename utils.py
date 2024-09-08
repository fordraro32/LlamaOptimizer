import random
import torch
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_optimizer(model, learning_rate=1e-5):
    return torch.optim.AdamW(model.parameters(), lr=learning_rate)

def get_scheduler(optimizer, num_training_steps, warmup_steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)

def format_metrics(metrics):
    return ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
