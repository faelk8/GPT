import torch
from metrics.metrics import calc_loss_batch, calc_loss_loader, calc_accuracy_loader


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Avalia o desempenho do modelo calculando a perda média tanto no conjunto de treino 
    quanto no conjunto de validação.

    Parâmetros:
    -----------
    model : torch.nn.Module
        O modelo PyTorch a ser avaliado.
    train_loader : DataLoader
        Loader contendo os dados de treino.
    val_loader : DataLoader
        Loader contendo os dados de validação.
    device : torch.device
        O dispositivo (CPU ou GPU) onde o modelo e os dados estão alocados.
    eval_iter : int
        Número de batches a serem avaliados para cálculo da perda em cada loader.

    Retorna:
    --------
    train_loss : float
        Perda média calculada no conjunto de treino.
    val_loss : float
        Perda média calculada no conjunto de validação.
    """
    # Coloca o modelo em modo avaliação (desliga dropout e batchnorm em modo treino)
    model.eval()

    # Desabilita o cálculo de gradiente para economizar memória e acelerar
    with torch.no_grad():
        # Calcula a perda média no conjunto de treino para 'eval_iter' batches
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter)

        # Calcula a perda média no conjunto de validação para 'eval_iter' batches
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter)

    # Retorna o modelo para modo treino após avaliação
    model.train()

    # Retorna as perdas médias calculadas
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    """
    Treina um modelo de classificação binária simples usando dados de treino e validação.

    Parâmetros:
    -----------
    model : torch.nn.Module
        Modelo a ser treinado.
    train_loader : DataLoader
        Loader contendo os dados de treino.
    val_loader : DataLoader
        Loader contendo os dados de validação.
    optimizer : torch.optim.Optimizer
        Otimizador para atualizar os pesos do modelo.
    device : torch.device
        Dispositivo onde os dados e modelo estão alocados (CPU ou GPU).
    num_epochs : int
        Número de épocas para o treinamento.
    eval_freq : int
        Frequência (em número de batches) para realizar avaliação durante o treinamento.
    eval_iter : int
        Número de batches usados para avaliação (cálculo de perda e acurácia).

    Retorna:
    --------
    train_losses : list de float
        Lista com as perdas no conjunto de treino avaliadas periodicamente.
    val_losses : list de float
        Lista com as perdas no conjunto de validação avaliadas periodicamente.
    train_accs : list de float
        Lista com as acurácias no conjunto de treino após cada época.
    val_accs : list de float
        Lista com as acurácias no conjunto de validação após cada época.
    examples_seen : int
        Número total de exemplos vistos durante o treinamento.
    """

    # Inicializa listas para armazenar perdas e acurácias durante o treinamento
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # Contadores para o número de exemplos processados e passos globais
    examples_seen, global_step = 0, -1

    # Loop principal de treinamento por época
    for epoch in range(num_epochs):
        model.train()  # Coloca o modelo em modo treino (ativa dropout, batchnorm, etc.)

        # Itera sobre batches do loader de treino
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Zera os gradientes acumulados do batch anterior

            # Calcula a perda do batch atual
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            loss.backward()  # Calcula os gradientes via backpropagation

            optimizer.step()  # Atualiza os pesos do modelo com base nos gradientes calculados

            # Atualiza o contador de exemplos vistos (baseado no tamanho do batch)
            examples_seen += input_batch.shape[0]
            # Incrementa o contador global de passos
            global_step += 1

            # Avaliação opcional periódica com base na frequência definida
            if global_step % eval_freq == 0:
                # Avalia perda média nos conjuntos de treino e validação
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                # Armazena as perdas para visualização posterior
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Após cada época, calcula a acurácia em treino e validação usando um número fixo de batches
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter)

        # Imprime acurácia no treino e validação
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")

        # Armazena as acurácias para visualização posterior
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    # Retorna as listas de perdas, acurácias e o total de exemplos vistos
    return train_losses, val_losses, train_accs, val_accs, examples_seen
