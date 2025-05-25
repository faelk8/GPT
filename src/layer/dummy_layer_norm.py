import torch.nn as nn


class DummyLayerNorm(nn.Module):
    """
    Implementação fictícia (dummy) de uma camada de normalização (LayerNorm).

    Esta classe serve como um substituto temporário para a camada de `LayerNorm`,
    útil em testes, benchmarks ou para desativar normalização sem alterar a estrutura
    do modelo.

    Parâmetros:
    -----------
    normalized_shape : int ou tuple
        Dimensão esperada dos vetores a serem normalizados (não utilizada aqui).
    eps : float, opcional (default=1e-5)
        Pequeno valor constante para estabilidade numérica (não utilizado aqui).
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()  # Inicializa a classe base nn.Module
        # Este é um placeholder; o valor de entrada é ignorado.
        pass  # Nenhuma operação é realizada

    def forward(self, x):
        """
        Executa o passo de inferência.

        Neste caso, simplesmente retorna a entrada sem qualquer modificação,
        simulando o comportamento de uma normalização desativada.

        Parâmetros:
        -----------
        x : torch.Tensor
            Entrada do modelo com qualquer forma.

        Retorna:
        --------
        x : torch.Tensor
            A mesma entrada, sem alterações.
        """
        # Apenas retorna a entrada sem nenhuma modificação
        return x
