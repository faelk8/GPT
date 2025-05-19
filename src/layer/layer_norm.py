import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        # Pequeno valor para evitar divisão por zero durante normalização
        self.eps = 1e-5

        # Parâmetro treinável para escalonar (gamma)
        self.scale = nn.Parameter(torch.ones(emb_dim))

        # Parâmetro treinável para deslocar (beta)
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Calcula a média por vetor de embedding (última dimensão)
        mean = x.mean(dim=-1, keepdim=True)

        # Calcula a variância (sem correção de Bessel — ou seja, biased)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normaliza: subtrai a média e divide pelo desvio padrão
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Aplica transformação aprendida (gamma * norm + beta)
        return self.scale * norm_x + self.shift
