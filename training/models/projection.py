import torch
import torch.nn as nn


class ImageProjector(nn.Module):
    def __init__(self, latent_dim: int = 1280, embed_dim: int = 1024):
        super().__init__()
        self.projection = nn.Linear(latent_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1280, h, w] -> global avg pool -> [batch, 1280] -> project
        pooled = torch.mean(x, dim=[2, 3])
        return self.projection(pooled)
