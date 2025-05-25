import torch.nn as nn

from attention.multi_head_attention import MultiHeadAttention
from layer.feed_forward import FeedForward
from layer.layer_norm import LayerNorm


class TransformerBlock(nn.Module):
    """
    Implementa um bloco do Transformer conforme descrito no paper "Attention is All You Need".

    Este bloco é composto por:
    - Atenção multi-cabeças com normalização e conexão residual (pré-norm)
    - Feed-forward (MLP) com normalização e conexão residual
    - Dropout aplicado após atenção e MLP

    Parâmetros:
    - cfg (dict): Dicionário de configuração contendo:
        - "emb_dim": dimensão dos embeddings
        - "context_length": tamanho da sequência (janela de atenção)
        - "n_heads": número de cabeças de atenção
        - "drop_rate": taxa de dropout
        - "qkv_bias": se deve usar bias nas projeções QKV

    Métodos:
    - forward(x): Processa uma sequência de embeddings com atenção + MLP com skip connections.

    Exemplo:
    >>> block = TransformerBlock(cfg)
    >>> x = torch.randn(batch_size, seq_len, emb_dim)
    >>> y = block(x)
    """

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
        """
        Executa o bloco Transformer com atenção e MLP.

        Parâmetros:
        - x (torch.Tensor): Tensor de entrada com forma [batch_size, seq_len, emb_dim]

        Retorna:
        - torch.Tensor: Tensor transformado com mesma forma da entrada.
        """
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
