import matplotlib.pyplot as plt


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
    fig, ax1 = plt.subplots(figsize=(5, 3))

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
