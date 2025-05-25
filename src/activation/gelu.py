import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    Implementa a função de ativação GELU (Gaussian Error Linear Unit).

    Esta função é uma alternativa suave à ReLU, usada frequentemente em Transformers
    e outros modelos de linguagem. Ela é definida por:

        GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    A versão aqui implementada é a aproximação usada pelo paper original do GPT.

    Métodos:
    - forward(x): Aplica a ativação GELU ao tensor de entrada.

    Exemplo:
    >>> gelu = GELU()
    >>> x = torch.randn(2, 3)
    >>> y = gelu(x)
    >>> y.shape
    torch.Size([2, 3])
    """

    def __init__(self):
        super().__init__()  # Inicializa a classe base nn.Module

    def forward(self, x):
        """
        Aplica a ativação GELU aproximada ao tensor de entrada.

        Parâmetros:
        - x (torch.Tensor): Tensor de entrada.

        Retorna:
        - torch.Tensor: Tensor com GELU aplicado elemento a elemento.
        """
        return 0.5 * x * (1 + torch.tanh(              # Fórmula aproximada da GELU
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *  # Constante √(2/π)
            # Termo cúbico para aproximação
            (x + 0.044715 * torch.pow(x, 3))
        ))
