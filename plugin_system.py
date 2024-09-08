from abc import ABC, abstractmethod

class OptimizationPlugin(ABC):
    @abstractmethod
    def optimize(self, model):
        pass

class PluginManager:
    def __init__(self):
        self.plugins = {}

    def register_plugin(self, name, plugin):
        if not isinstance(plugin, OptimizationPlugin):
            raise ValueError("Plugin must be an instance of OptimizationPlugin")
        self.plugins[name] = plugin

    def apply_plugin(self, name, model):
        if name not in self.plugins:
            raise ValueError(f"Plugin '{name}' not found")
        return self.plugins[name].optimize(model)

    def apply_all_plugins(self, model):
        for name, plugin in self.plugins.items():
            model = plugin.optimize(model)
        return model

# Example plugin
class GradientClippingPlugin(OptimizationPlugin):
    def __init__(self, max_norm):
        self.max_norm = max_norm

    def optimize(self, model):
        for param in model.parameters():
            torch.nn.utils.clip_grad_norm_(param, self.max_norm)
        return model
