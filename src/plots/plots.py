import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    """
    Plota os valores de treino e validação ao longo das épocas, com um segundo eixo indicando exemplos vistos.

    Parâmetros:
    -----------
    epochs_seen : list or array-like
        Lista com os valores de época correspondentes a cada ponto.

    examples_seen : list or array-like
        Lista com o número acumulado de exemplos processados ao longo do tempo.

    train_values : list or array-like
        Valores da métrica (ex: perda) para o conjunto de treinamento.

    val_values : list or array-like
        Valores da métrica (ex: perda) para o conjunto de validação.

    label : str, default="loss"
        Nome da métrica a ser usada nos rótulos e no nome do arquivo salvo.

    Salva:
    ------
    - Um arquivo PDF com o gráfico no formato '{label}-plot.pdf'.
    - Exibe o gráfico em tela.
    """

    # Cria figura e eixo principal (para epochs)
    fig, ax1 = plt.subplots(figsize=(18, 6))

    # Plota os valores de treino e validação em função das épocas
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.",
             label=f"Validation {label}")
    ax1.set_xlabel("Epochs")  # Rótulo do eixo x principal
    ax1.set_ylabel(label.capitalize())  # Rótulo do eixo y com capitalização
    ax1.legend()  # Mostra a legenda com os rótulos das curvas

    # Cria um segundo eixo x (compartilha o eixo y)
    ax2 = ax1.twiny()

    # Plota algo invisível apenas para alinhar os ticks do segundo eixo x
    ax2.plot(examples_seen, train_values, alpha=0)

    # Rótulo do segundo eixo x
    ax2.set_xlabel("Examples seen")

    # Ajusta o layout para evitar sobreposição de elementos
    fig.tight_layout()

    # Salva o gráfico em um arquivo PDF
    plt.savefig(f"{label}-plot.pdf")

    # Exibe o gráfico
    plt.show()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plota os valores de perda (loss) de treino e validação ao longo das épocas, 
    com um segundo eixo x indicando o número de tokens processados.

    Parâmetros:
    -----------
    epochs_seen : list[int]
        Lista contendo os números de época (epoch) correspondentes aos valores de perda.

    tokens_seen : list[int]
        Lista contendo o número acumulado de tokens processados ao longo do treinamento.

    train_losses : list[float]
        Lista de valores de perda do conjunto de treinamento para cada época.

    val_losses : list[float]
        Lista de valores de perda do conjunto de validação para cada época.
    """

    # Cria a figura e os eixos principais (y = loss, x = epochs)
    fig, ax1 = plt.subplots(figsize=(18, 6))

    # Plota a curva de perda de treinamento
    ax1.plot(epochs_seen, train_losses, label="Training loss")

    # Plota a curva de perda de validação com linha tracejada
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")

    # Define o rótulo do eixo x primário como "Epochs"
    ax1.set_xlabel("Epochs")

    # Define o rótulo do eixo y como "Loss"
    ax1.set_ylabel("Loss")

    # Exibe a legenda no canto superior direito
    ax1.legend(loc="upper right")

    # Mostra apenas rótulos inteiros no eixo x (útil para epochs)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Cria um segundo eixo x no topo do gráfico para tokens vistos
    ax2 = ax1.twiny()  # Compartilha o mesmo eixo y

    # Plotagem invisível para alinhar o eixo de tokens com a curva de perdas
    ax2.plot(tokens_seen, train_losses, alpha=0)

    # Define o rótulo do segundo eixo x como "Tokens seen"
    ax2.set_xlabel("Tokens seen")

    # Ajusta automaticamente o layout da figura para evitar sobreposição de elementos
    fig.tight_layout()

    # Exibe o gráfico
    plt.show()
