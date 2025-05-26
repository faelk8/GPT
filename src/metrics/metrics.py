import torch
import torch.nn as nn


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    Calcula a acurácia do modelo com base nos dados fornecidos pelo DataLoader.

    Parâmetros:
    -----------
    data_loader : DataLoader
        Objeto do PyTorch que fornece lotes de dados (entrada e rótulo alvo).

    model : torch.nn.Module
        Modelo de linguagem ou classificação a ser avaliado.

    device : torch.device
        Dispositivo onde os tensores e o modelo devem ser colocados (CPU ou CUDA).

    num_batches : int, opcional
        Número máximo de lotes a serem processados. Se None, processa todos os lotes.

    Retorna:
    --------
    float
        Acurácia do modelo no conjunto avaliado.
    """
    model.eval()  # Coloca o modelo em modo de avaliação
    correct_predictions, num_examples = 0, 0  # Inicializa contadores

    # Define quantos lotes serão avaliados
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # Move os dados para o dispositivo
            input_batch, target_batch = input_batch.to(
                device), target_batch.to(device)

            with torch.no_grad():  # Desativa o cálculo de gradientes
                # Pega os logits do último token
                logits = model(input_batch)[:, -1, :]

            # Pega a classe predita
            predicted_labels = torch.argmax(logits, dim=-1)

            # Atualiza os contadores
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels ==
                                    target_batch).sum().item()
        else:
            break

    return correct_predictions / num_examples  # Retorna a acurácia


def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calcula a perda de entropia cruzada para um único lote de dados.

    Parâmetros:
    -----------
    input_batch : torch.Tensor
        Lote de entrada (sequências tokenizadas).

    target_batch : torch.Tensor
        Lote de rótulos (tokens esperados).

    model : torch.nn.Module
        Modelo que realiza a predição.

    device : torch.device
        Dispositivo onde os tensores e o modelo devem estar (CPU ou CUDA).

    Retorna:
    --------
    torch.Tensor
        Valor escalar da perda (loss) para o lote.
    """
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)[:, -1, :]  # Pega os logits do último token
    loss = torch.nn.functional.cross_entropy(
        logits, target_batch)  # Calcula a perda
    return loss


def calc_loss_batch_instruction(input_batch, target_batch, model, device):
    """
    Calcula a perda (loss) entre os logits produzidos pelo modelo e os rótulos-alvo (target_batch),
    usando entropia cruzada (cross-entropy), apropriada para tarefas de linguagem com tokenização.

    Este método assume que `input_batch` e `target_batch` estão no formato (batch_size, seq_len),
    e que o modelo retorna logits no formato (batch_size, seq_len, vocab_size).

    Parâmetros:
    -----------
    input_batch : torch.Tensor
        Tensor de entrada com os IDs dos tokens para cada sequência do batch.

    target_batch : torch.Tensor
        Tensor de rótulos esperados, normalmente a mesma sequência deslocada (para modelos autoregressivos).

    model : torch.nn.Module
        O modelo de linguagem que será utilizado para gerar os logits.

    device : str ou torch.device
        O dispositivo no qual os tensores e o modelo devem estar (por exemplo: "cuda" ou "cpu").

    Retorna:
    --------
    loss : torch.Tensor
        A perda calculada para o batch inteiro (scalar).
    """

    # Move os tensores de entrada e alvo para o dispositivo especificado (CPU ou GPU)
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # Passa o input pelo modelo e obtém os logits (probabilidades antes do softmax)
    # Saída esperada: (batch_size, seq_len, vocab_size)
    logits = model(input_batch)

    # Reorganiza os tensores para o formato esperado pela função de perda
    # logits: (batch_size * seq_len, vocab_size)
    # targets: (batch_size * seq_len)
    loss = torch.nn.functional.cross_entropy(
        # Combina batch_size e seq_len em uma só dimensão
        logits.flatten(0, 1),
        target_batch.flatten()        # Alinha os rótulos com a nova dimensão
    )

    # Retorna a perda escalar
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calcula a perda média (cross-entropy) do modelo sobre um DataLoader.

    Parâmetros:
    -----------
    data_loader : DataLoader
        Objeto do PyTorch que fornece lotes de entrada e saída esperada.

    model : torch.nn.Module
        Modelo de linguagem ou classificação a ser avaliado.

    device : torch.device
        Dispositivo de execução (CPU ou CUDA).

    num_batches : int, opcional
        Número máximo de lotes a serem processados. Se None, processa todos os lotes.

    Retorna:
    --------
    float
        Perda média do modelo nos lotes avaliados. Retorna NaN se o DataLoader estiver vazio.
    """
    total_loss = 0.

    if len(data_loader) == 0:
        return float("nan")  # Retorna NaN se não há dados

    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Garante que não ultrapasse o tamanho real
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch_instruction(
                input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches  # Retorna a perda média


def calc_loss_loader_instruction(data_loader, model, device, num_batches=None):
    """
    Calcula a perda média de um modelo sobre os batches de um DataLoader.

    Esta função itera sobre um número limitado (ou total) de batches do DataLoader,
    calcula a perda de cada batch usando a função `calc_loss_batch_instruction`,
    e retorna a média dessas perdas.

    Parâmetros:
    -----------
    data_loader : torch.utils.data.DataLoader
        DataLoader contendo batches de (input_batch, target_batch).

    model : torch.nn.Module
        O modelo de linguagem treinado ou em avaliação.

    device : str ou torch.device
        O dispositivo onde os dados e o modelo estão localizados ("cuda" ou "cpu").

    num_batches : int ou None
        Número de batches a considerar. Se None, será usado o total de batches do DataLoader.

    Retorna:
    --------
    float
        A perda média (cross-entropy) nos batches avaliados.
    """

    total_loss = 0.0  # Acumulador da perda total

    # Se o DataLoader estiver vazio, retorna NaN
    if len(data_loader) == 0:
        return float("nan")

    # Se não foi especificado num_batches, use o tamanho total do DataLoader
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Garante que num_batches não ultrapasse o total de batches no DataLoader
        num_batches = min(num_batches, len(data_loader))

    # Loop pelos batches
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # Calcula a perda para o batch atual
            loss = calc_loss_batch_instruction(
                input_batch, target_batch, model, device)
            # Soma o valor escalar da perda ao total acumulado
            total_loss += loss.item()
        else:
            # Para o loop se já atingiu o número de batches desejado
            break

    # Retorna a média da perda total
    return total_loss / num_batches


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """
    Calcula a acurácia do modelo com base nas predições feitas em lotes fornecidos por um DataLoader.

    Parâmetros:
    -----------
    data_loader : torch.utils.data.DataLoader
        DataLoader contendo os pares (entrada, alvo) para avaliação.

    model : torch.nn.Module
        Modelo treinado que será avaliado.

    device : torch.device
        Dispositivo no qual os dados e o modelo serão movidos (ex: torch.device("cuda") ou torch.device("cpu")).

    num_batches : int, opcional
        Número máximo de batches a serem avaliados. Se None, todos os batches serão utilizados.

    Retorna:
    --------
    float
        Acurácia média (entre 0 e 1) calculada nos batches avaliados.
    """
    model.eval()  # Coloca o modelo em modo de avaliação (desativa dropout, etc.)
    correct_predictions, num_examples = 0, 0  # Inicializa os contadores

    # Define o número real de batches a serem processados
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    # Loop pelos batches do DataLoader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # Move os dados para o dispositivo (GPU ou CPU)
            input_batch, target_batch = input_batch.to(
                device), target_batch.to(device)

            with torch.no_grad():  # Desativa cálculo de gradiente (mais eficiente para inferência)
                # Obtém os logits do último token da sequência
                logits = model(input_batch)[:, -1, :]

            # Converte logits em rótulos preditos
            predicted_labels = torch.argmax(logits, dim=-1)

            # Atualiza o número total de exemplos e o número de acertos
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels ==
                                    target_batch).sum().item()
        else:
            break  # Interrompe se já processou o número desejado de batches

    # Retorna a acurácia como proporção de acertos
    return correct_predictions / num_examples
