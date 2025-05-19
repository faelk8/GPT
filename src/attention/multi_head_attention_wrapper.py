import torch
import torch.nn as nn

from causal_attention import CausalAttention


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # Cria uma lista de cabeças de atenção (CausalAttention)
        # Cada cabeça é independente e aprende diferentes representações
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
             for _ in range(num_heads)]
        )

    def forward(self, x):
        # Executa cada cabeça de atenção e concatena os resultados ao longo da última dimensão (d_out)
        return torch.cat([head(x) for head in self.heads], dim=-1)
