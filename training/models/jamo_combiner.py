"""Korean Jamo Combiner: 3 jamo embeddings → 1 character embedding via Linear."""

import torch
import torch.nn as nn


class JamoCombiner(nn.Module):
    """Combine up to 3 jamo token embeddings into a single character embedding.

    Input:  [batch, num_chars, 3, embed_dim]  (초성, 중성, 종성)
    Output: [batch, num_chars, embed_dim]
    """

    def __init__(self, embed_dim: int, num_jamo_slots: int = 3):
        super().__init__()
        self.linear = nn.Linear(embed_dim * num_jamo_slots, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, num_chars, 3, embed_dim]
        b, n, s, d = x.shape
        x = x.view(b, n, s * d)  # [batch, num_chars, 3*embed_dim]
        return self.linear(x)    # [batch, num_chars, embed_dim]
