import torch
import torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

class FlexibleModel(nn.Module):
    def __init__(self,
                 d_model=200,
                 d_state=16,
                 d_conv=4,
                 expand=2,
                 num_layers=4,
                 dropout=0.1,
                 use_residual=True,
                 fusion_mode='interleave',
                 pooling='gap',
                 classifier_depth=3,
                 **kwargs):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.pooling = pooling
        self.use_residual = use_residual

        self.backbone = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            self.backbone.append(
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
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
            self.norms.append(nn.LayerNorm(d_model))

        if classifier_depth == 1:
            self.classifier = nn.Linear(d_model, 1)
        elif classifier_depth == 2:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1),
            )

    def _apply_backbone(self, x):
        for layer, norm in zip(self.backbone, self.norms):
            if self.use_residual:
                residual = x
                x = layer(x)
                x = norm(x + residual)
            else:
                x = layer(x)
                x = norm(x)
        return x

    def forward(self, x):
        b, c, t, d = x.shape

        if self.fusion_mode == 'interleave':
            x = x.view(b, c * t, d)
            x = self._apply_backbone(x)

        elif self.fusion_mode == 'concat_feat':
            x = x.permute(0, 2, 1, 3).reshape(b, t, c * d)
            x = self._apply_backbone(x)

        elif self.fusion_mode == 'parallel':
            x_ch1 = x[:, 0, :, :]
            x_ch2 = x[:, 1, :, :]
            x_ch1 = self._apply_backbone(x_ch1)
            x_ch2 = self._apply_backbone(x_ch2)
            x = torch.cat([x_ch1, x_ch2], dim=-1)
            x = x.mean(dim=1)
            return self.classifier(x).squeeze(-1)

        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

        if self.pooling == 'gap':
            x = x.mean(dim=1)
        elif self.pooling == 'last':
            x = x[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        logits = self.classifier(x)
        return logits.squeeze(-1)