import torch.nn as nn

from attention.multi_head_attention import MultiHeadAttention
from layer.feed_forward import FeedForward
from layer.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Módulo de atenção multi-cabeças
        self.att = MultiHeadAttention(
            # Dimensão de entrada dos embeddings
            d_in=cfg["emb_dim"],
            # Dimensão de saída (mesmo tamanho para residual)
            d_out=cfg["emb_dim"],
            # Tamanho do contexto (número de tokens)
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],          # Número de cabeças de atenção
            # Taxa de dropout aplicada na atenção
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])          # Define se bias será usado nas projeções QKV

        # Módulo feed-forward (MLP posicional)
        self.ff = FeedForward(cfg)

        # Normalizações aplicadas antes da atenção e do feed-forward (pré-norm)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout aplicado após cada bloco (atenção e FF)
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # === Bloco de Atenção com conexão residual ===
        shortcut = x                  # Guarda a entrada para conexão residual
        x = self.norm1(x)             # Normalização antes da atenção
        x = self.att(x)               # Atenção multi-cabeças
        x = self.drop_shortcut(x)     # Dropout após a atenção
        x = x + shortcut              # Soma residual (skip connection)

        # === Bloco Feed-Forward com conexão residual ===
        shortcut = x                  # Atualiza o shortcut
        x = self.norm2(x)             # Normalização antes do MLP
        x = self.ff(x)                # Feed-forward (MLP)
        x = self.drop_shortcut(x)     # Dropout após o MLP
        x = x + shortcut              # Soma residual

        return x
