import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

class Model(nn.Module):
    def __init__(self, param):
        super().__init__()
        self.backbone = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(4):
            self.backbone.append(
                Mamba(
                    d_model=200,
                    d_state=16,
                    d_conv=4,
                    expand=2,
                    dt_rank="auto",
                    dt_min=0.001,
                    dt_max=0.1,
                    dt_init="random",
                    dt_scale=1.0,
                    dt_init_floor=1e-4,
                    conv_bias=True,
                    bias=False,
                    use_fast_path=True,
                    layer_idx=i,
                )
            )
            self.norms.append(nn.LayerNorm(200))

        self.classifier = nn.Sequential(
            nn.Linear(200, 128),
            nn.ReLU(),
            nn.Dropout(param.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(param.dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        # x shape: (B, C, T, D) with C=2, T=30, D=200
        b, c, t, d = x.shape
        x = x.view(b, c * t, d)      # (B, 60, 200)

        for layer, norm in zip(self.backbone, self.norms):
            residual = x
            x = layer(x)
            x = norm(x + residual)

        x = x.mean(dim=1)            # global average pooling -> (B, 200)
        logits = self.classifier(x)  # (B, 1)
        return logits.squeeze(-1)    # (B,)