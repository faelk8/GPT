import torch
from metrics.metrics import calc_loss_batch, calc_loss_loader, calc_loss_loader_instruction, calc_accuracy_loader, calc_loss_batch_instruction
from function.geral import text_to_token_ids, token_ids_to_text, generate_text_simple


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


def evaluate_model_instruction(model, train_loader, val_loader, device, eval_iter):
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
        train_loss = calc_loss_loader_instruction(
            train_loader, model, device, num_batches=eval_iter)

        # Calcula a perda média no conjunto de validação para 'eval_iter' batches
        val_loss = calc_loss_loader_instruction(
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


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """
    Treina um modelo de linguagem de forma simples com avaliação periódica.

    Parâmetros:
    -----------
    model : torch.nn.Module
        O modelo a ser treinado.

    train_loader : DataLoader
        DataLoader contendo os dados de treinamento (pares input-target).

    val_loader : DataLoader
        DataLoader contendo os dados de validação.

    optimizer : torch.optim.Optimizer
        Otimizador utilizado para atualizar os pesos do modelo.

    device : str ou torch.device
        Dispositivo onde os dados e o modelo serão alocados (ex: "cuda" ou "cpu").

    num_epochs : int
        Número total de épocas de treinamento.

    eval_freq : int
        Frequência (em número de batches) com que o modelo será avaliado.

    eval_iter : int
        Número de batches a serem usados na avaliação para calcular a loss média.

    start_context : str
        Contexto inicial (prompt) para gerar e imprimir uma amostra após cada época.

    tokenizer : Tokenizer
        Tokenizador usado para decodificar os resultados da geração.

    Retorna:
    --------
    train_losses : list[float]
        Lista contendo as perdas de treino ao longo do treinamento.

    val_losses : list[float]
        Lista contendo as perdas de validação ao longo do treinamento.

    track_tokens_seen : list[int]
        Lista com o número de tokens vistos nos momentos de avaliação.
    """

    # Inicializa listas para armazenar perdas e tokens vistos
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1  # contador de tokens e passos globais

    # Loop principal de treinamento por época
    for epoch in range(num_epochs):
        model.train()  # Coloca o modelo em modo de treinamento

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Zera os gradientes do otimizador

            # Calcula a perda para o batch atual
            loss = calc_loss_batch_instruction(
                input_batch, target_batch, model, device)

            # Backpropagation: calcula os gradientes
            loss.backward()

            # Atualiza os pesos do modelo com os gradientes calculados
            optimizer.step()

            # Atualiza contadores
            tokens_seen += input_batch.numel()  # número de tokens processados
            global_step += 1  # incremento do passo global

            # Avaliação periódica do modelo
            if global_step % eval_freq == 0:
                # Calcula perda média em parte do dataset
                train_loss, val_loss = evaluate_model_instruction(
                    model, train_loader, val_loader, device, eval_iter)

                # Armazena os valores
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)

                # Exibe métricas
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Geração de texto de amostra após cada época
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    # Retorna métricas rastreadas
    return train_losses, val_losses, track_tokens_seen


def generate_and_print_sample(model, tokenizer, device, start_context):
    """
    Gera uma amostra de texto a partir de um contexto inicial usando o modelo treinado
    e imprime o resultado em formato compacto.

    Parâmetros:
    -----------
    model : torch.nn.Module
        O modelo de linguagem treinado.

    tokenizer : Tokenizer
        Tokenizador usado para codificar o texto de entrada e decodificar a saída gerada.

    device : str ou torch.device
        Dispositivo onde os tensores e o modelo estão alocados.

    start_context : str
        Texto inicial (prompt) usado como entrada para geração de texto.
    """

    model.eval()  # Coloca o modelo em modo de avaliação (desativa dropout, etc.)

    # Obtém o tamanho do contexto máximo suportado pelo modelo (com base nos embeddings)
    context_size = model.pos_emb.weight.shape[0]

    # Codifica o texto de entrada em IDs de tokens e envia para o dispositivo
    encoded = text_to_token_ids(start_context, tokenizer).to(device)

    # Geração do texto com o modelo (sem calcular gradientes)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )

        # Decodifica os IDs de volta para texto legível
        decoded_text = token_ids_to_text(token_ids, tokenizer)

        # Imprime o texto gerado com quebras de linha substituídas por espaço
        # Saída mais compacta no terminal
        print(decoded_text.replace("\n", " "))

    model.train()  # Retorna o modelo para modo de treinamento
