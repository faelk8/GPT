import torch
import torch.nn as nn

from layer.gelu import GELU


class ExampleDeepNeuralNetwork(nn.Module):
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
