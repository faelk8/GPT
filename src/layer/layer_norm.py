import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Implementa a normalização por camada (Layer Normalization) manualmente,
    com parâmetros treináveis de escala e deslocamento.

    A normalização é feita sobre a última dimensão do tensor de entrada, o que é
    adequado para sequências de embeddings em modelos como Transformers.

    Parâmetros:
    -----------
    emb_dim : int
        A dimensão do embedding a ser normalizado. Define o tamanho dos vetores
        de escala (gamma) e deslocamento (beta).
    """

    def __init__(self, emb_dim):
        super().__init__()

        # Pequeno valor para evitar divisão por zero durante normalização
        self.eps = 1e-5

        # Parâmetro treinável para escalonar (gamma)
        self.scale = nn.Parameter(torch.ones(emb_dim))

        # Parâmetro treinável para deslocar (beta)
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        """
        Aplica a normalização por camada ao tensor de entrada.

        A normalização é feita subtraindo a média e dividindo pelo desvio padrão
        da última dimensão do tensor, seguida pela aplicação dos parâmetros
        treináveis de escala e deslocamento.

        Parâmetros:
        -----------
        x : torch.Tensor
            Tensor de entrada com forma (batch_size, seq_len, emb_dim).

        Retorna:
        --------
        torch.Tensor
            Tensor normalizado com mesma forma da entrada.
        """
        # Calcula a média por vetor de embedding (última dimensão)
        mean = x.mean(dim=-1, keepdim=True)

        # Calcula a variância (sem correção de Bessel — ou seja, biased)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normaliza: subtrai a média e divide pelo desvio padrão
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Aplica transformação aprendida (gamma * norm + beta)
        return self.scale * norm_x + self.shift
