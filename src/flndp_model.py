import torch.nn as nn

class BreastCancerMLP(nn.Module):
    def __init__(self, arch_config: dict):
        super().__init__()
        layers = []
        for layer_cfg in arch_config["layers"]:
            layer_type = getattr(nn, layer_cfg.pop("type"))
            layers.append(layer_type(**layer_cfg))
        self.layers = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        return self.layers(x)
