import torch.nn as nn
from torch.nn import GELU


class FeedForward(nn.Module):
    """
    Implementa o bloco feedforward de um Transformer.

    Este bloco é composto por duas camadas lineares com uma ativação não linear GELU entre elas.
    A primeira camada expande a dimensionalidade da entrada em 4x, e a segunda reduz de volta ao original.

    Parâmetros:
    - cfg (dict): Um dicionário de configuração que deve conter:
        - "emb_dim" (int): Dimensão do embedding (entrada e saída do bloco).

    Métodos:
    - forward(x): Executa a passagem direta do tensor `x` pela sequência de camadas.

    Exemplo:
    >>> cfg = {"emb_dim": 768}
    >>> ff = FeedForward(cfg)
    >>> x = torch.randn(32, 10, 768)
    >>> y = ff(x)
    >>> y.shape
    torch.Size([32, 10, 768])
    """

    def __init__(self, cfg):
        super().__init__()  # Inicializa a classe base nn.Module

        # Define as camadas sequenciais do bloco feedforward:
        # 1. Linear: emb_dim → 4 * emb_dim
        # 2. Ativação: GELU (não linear)
        # 3. Linear: 4 * emb_dim → emb_dim
        self.layers = nn.Sequential(
            # Expansão da dimensão
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),  # Ativação GELU (Gaussian Error Linear Unit)
            # Redução para a dimensão original
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        """
        Aplica a sequência de camadas do feedforward ao tensor de entrada.

        Parâmetros:
        - x (torch.Tensor): Tensor de entrada com forma (batch_size, seq_len, emb_dim).

        Retorna:
        - torch.Tensor: Tensor processado com a mesma forma de entrada.
        """
        return self.layers(x)  # Aplica as camadas sequenciais ao tensor de entrada
