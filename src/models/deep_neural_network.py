import torch
import torch.nn as nn

from activation.gelu import GELU


class ExampleDeepNeuralNetwork(nn.Module):
    """
    Exemplo de rede neural profunda com múltiplas camadas lineares seguidas de ativação GELU.
    Permite a ativação opcional de conexões de atalho (residuais) entre as camadas para facilitar o treinamento.

    Parâmetros:
    -----------
    layer_sizes : list[int]
        Lista com os tamanhos das camadas, onde cada par consecutivo define a entrada e saída de cada camada Linear.
        Por exemplo, [input_dim, hidden1, hidden2, ..., output_dim].
    use_shortcut : bool
        Flag para ativar ou desativar conexões de atalho (residuais) entre as camadas.

    Método forward(x):
    -----------------
    Executa o forward pass da rede aplicando cada camada sequencialmente. Se use_shortcut estiver ativado
    e a entrada e saída da camada tiverem a mesma forma, o resultado da camada será somado à entrada (residual).

    Parâmetros:
    -----------
    x : torch.Tensor
        Tensor de entrada para a rede.

    Retorna:
    --------
    torch.Tensor
        Tensor resultante da passagem pela rede.
    """

    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        # Ativa ou desativa conexões de atalho (residuais)
        self.use_shortcut = use_shortcut

        # Cria uma lista de camadas sequenciais com Linear + GELU para cada par consecutivo de tamanhos
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Executa o forward da camada atual
            layer_output = layer(x)

            # Se ativado, e se o input e output tiverem o mesmo shape, aplica o shortcut (residual)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output  # Caso contrário, apenas passa o output da camada
        return x
