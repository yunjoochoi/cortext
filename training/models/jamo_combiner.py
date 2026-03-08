"""Korean Jamo Combiner: 3 jamo embeddings → 1 character embedding via Linear."""

import torch
import torch.nn as nn


class JamoCombiner(nn.Module):
    """Combine up to 3 jamo token embeddings into a single character embedding."""

    def __init__(self, embed_dim: int, num_jamo_slots: int = 3):
        super().__init__()
        self.linear = nn.Linear(embed_dim * num_jamo_slots, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, s, d = x.shape
        x = x.view(b, n, s * d)  
        return self.linear(x)    