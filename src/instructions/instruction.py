import torch


def format_input(entry):
    """
    Formata uma entrada do tipo dicionário contendo instruções e, opcionalmente, uma entrada adicional.

    A função é usada para criar uma string padronizada que combina a instrução e o input (se houver),
    no estilo típico de datasets utilizados para treinar modelos de linguagem, como o formato do Alpaca, InstructGPT, etc.

    Parâmetros:
    - entry (dict): Um dicionário com as chaves 'instruction' e 'input'.
        - 'instruction' (str): Texto descrevendo a tarefa a ser realizada.
        - 'input' (str ou vazio): Informação de entrada adicional, se existir.

    Retorna:
    - str: Uma string formatada que combina a instrução e o input (se houver), pronta para ser passada ao modelo.
    """

    # Cria a parte inicial do prompt com a instrução fornecida
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # Se houver conteúdo em 'input', adiciona a seção de input formatada; caso contrário, deixa em branco
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    # Retorna a concatenação da instrução com o input (se houver)
    return instruction_text + input_text


def custom_collate_draft_1(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    """
    Função personalizada de colagem (collate) para DataLoader do PyTorch.

    Essa função prepara um lote de sequências para entrada em modelos de linguagem, 
    adicionando token de finalização (<|endoftext|>) ao final de cada sequência 
    e realizando padding para igualar os comprimentos.

    Parâmetros:
    - batch (list[list[int]]): Lista de sequências de tokens (listas de inteiros).
    - pad_token_id (int): ID do token de padding (por padrão 50256, que é o <|endoftext|> do GPT-2).
    - device (str): Dispositivo onde o tensor final será alocado ("cpu" ou "cuda").

    Retorna:
    - torch.Tensor: Tensor 2D contendo as sequências padronizadas, preparadas para input no modelo.
    """

    # Encontra o comprimento da sequência mais longa no batch e adiciona +1
    # para inserir o token de fim <|endoftext|> posteriormente
    batch_max_length = max(len(item)+1 for item in batch)

    # Lista para armazenar os tensores processados
    inputs_lst = []

    for item in batch:
        # Cria uma cópia da sequência original
        new_item = item.copy()

        # Adiciona o token de finalização no final da sequência
        new_item += [pad_token_id]

        # Realiza padding para que todas as sequências fiquem com o mesmo comprimento
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )

        # Remove o último token da sequência (o padding extra),
        # ele será usado em outro momento (por exemplo, como target para previsão)
        inputs = torch.tensor(padded[:-1])

        # Adiciona o tensor processado à lista
        inputs_lst.append(inputs)

    # Empilha todos os tensores em um único tensor e envia para o dispositivo desejado
    inputs_tensor = torch.stack(inputs_lst).to(device)

    # Retorna o tensor final do batch
    return inputs_tensor


def custom_collate_draft_2(
    batch,
    pad_token_id=50256,
    device="cpu"
):
    """
    Prepara um batch de sequências para entrada em um modelo de linguagem.

    Essa função realiza o padding das sequências do batch com um token de finalização,
    garante que todas tenham o mesmo comprimento, e prepara os tensores de `inputs` e `targets`
    de forma que possam ser usados para treino com shift de uma posição.

    Parâmetros:
    ----------
    batch : List[List[int]]
        Lista de sequências (cada uma é uma lista de IDs de tokens).
    pad_token_id : int, opcional
        O ID do token usado para preenchimento (padding). Default é 50256.
    device : str, opcional
        O dispositivo onde os tensores devem ser alocados ('cpu' ou 'cuda'). Default é 'cpu'.

    Retorna:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Um par de tensores:
        - inputs: tensor com as sequências de entrada (sem o último token).
        - targets: tensor com as sequências-alvo (sem o primeiro token), com shift de +1.
    """

    # Encontra o comprimento da maior sequência no batch, somando 1 para o token final (<|endoftext|>)
    batch_max_length = max(len(item) + 1 for item in batch)

    # Inicializa listas para armazenar tensores de entrada e alvo
    inputs_lst, targets_lst = [], []

    # Itera sobre cada sequência no batch
    for item in batch:
        # Cria uma cópia da sequência original para evitar modificar o original
        new_item = item.copy()

        # Adiciona o token de finalização (<|endoftext|>)
        new_item += [pad_token_id]

        # Preenche a sequência com tokens de padding até atingir o comprimento máximo
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )

        # Cria os tensores:
        # inputs: a sequência sem o último token
        inputs = torch.tensor(padded[:-1])
        # targets: a sequência sem o primeiro token (shift de +1)
        targets = torch.tensor(padded[1:])

        # Adiciona à lista de inputs e targets
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Empilha todos os tensores de input e move para o dispositivo especificado
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    # Retorna os tensores de inputs e targets prontos para o modelo
    return inputs_tensor, targets_tensor


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Prepara um batch de sequências para treino de modelos de linguagem, com padding,
    shift nos alvos (targets), substituição condicional por `ignore_index`, e
    truncamento opcional do comprimento.

    Essa função:
    - Adiciona um token <|endoftext|> (ou similar) ao final de cada sequência.
    - Faz padding até o comprimento máximo do batch.
    - Gera pares de inputs e targets com deslocamento de 1 posição (shift).
    - Substitui os tokens de padding em `targets` por `ignore_index`, exceto o primeiro.
    - Trunca as sequências, se necessário.
    - Retorna tensores prontos para uso em modelos de linguagem autoregressivos.

    Parâmetros:
    ----------
    batch : List[List[int]]
        Lista de sequências de tokens (listas de inteiros).
    pad_token_id : int, opcional
        Token utilizado para padding e fim de sequência. Default: 50256.
    ignore_index : int, opcional
        Valor usado para ignorar tokens no cálculo da loss (usado em `targets`). Default: -100.
    allowed_max_length : int ou None, opcional
        Limite superior de comprimento para truncamento das sequências. Default: None.
    device : str, opcional
        Dispositivo onde os tensores serão alocados ('cpu' ou 'cuda'). Default: 'cpu'.

    Retorna:
    -------
    Tuple[torch.Tensor, torch.Tensor]
        inputs_tensor: Tensor com sequências de entrada (sem último token, possivelmente truncadas).
        targets_tensor: Tensor com sequências de saída com shift +1 e padding mascarado com `ignore_index`.
    """

    # Encontra o comprimento da maior sequência no batch (adicionando 1 pelo <|endoftext|>)
    batch_max_length = max(len(item) + 1 for item in batch)

    # Listas para armazenar os tensores individuais de entrada e alvo
    inputs_lst, targets_lst = [], []

    # Itera por cada item (sequência) do batch
    for item in batch:
        # Copia a sequência para não modificar a original
        new_item = item.copy()

        # Adiciona o token de fim de sequência (<|endoftext|>)
        new_item += [pad_token_id]

        # Adiciona padding ao final para igualar ao comprimento máximo
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )

        # Cria o tensor de entrada removendo o último token (shift para esquerda)
        inputs = torch.tensor(padded[:-1])

        # Cria o tensor de target removendo o primeiro token (shift para direita)
        targets = torch.tensor(padded[1:])

        # Encontra os índices de tokens de padding no target
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        # Se houver mais de um padding, ignora todos exceto o primeiro
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Se um comprimento máximo for definido, trunca inputs e targets
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        # Adiciona os tensores à lista
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Empilha as listas em tensores 2D e move para o dispositivo desejado
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    # Retorna os tensores prontos para o modelo
    return inputs_tensor, targets_tensor
