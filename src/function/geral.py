import os
import torch
import tiktoken
import pandas as pd
import urllib.request
from torch.utils.data import DataLoader

from tk.gpt_datasetv1 import GPTDatasetV1


def load_text_file_from_url_if_needed(file_path, url):
    """
    Verifica se o arquivo existe localmente. Se não existir, faz download do conteúdo a partir da URL fornecida,
    salva no caminho especificado e retorna o conteúdo. Se já existir, apenas lê e retorna o conteúdo do arquivo.

    Args:
        file_path (str): Caminho local onde o arquivo deve ser salvo/lido.
        url (str): URL de onde baixar o arquivo, caso ele não exista.

    Returns:
        str: Conteúdo do arquivo de texto.
    """
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
    return text_data


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    Gera uma sequência de texto a partir de um modelo autoregressivo (como GPT).

    Parâmetros:
    -----------
    model : torch.nn.Module
        O modelo treinado que irá gerar os próximos tokens com base no contexto atual.

    idx : torch.Tensor
        Tensor de entrada com índices dos tokens já conhecidos. Espera-se que tenha shape (B, T),
        onde B é o batch size (geralmente 1) e T é o número de tokens no contexto.

    max_new_tokens : int
        Número máximo de novos tokens a serem gerados.

    context_size : int
        Tamanho máximo do contexto aceito pelo modelo (ex: 1024 para GPT-2).
        Se o número de tokens em `idx` exceder esse valor, apenas os últimos `context_size`
        tokens serão usados.

    Retorno:
    --------
    idx : torch.Tensor
        Tensor contendo os tokens originais e os novos tokens gerados.
    """
    # Loop para gerar até `max_new_tokens` novos tokens
    for _ in range(max_new_tokens):

        # Se o contexto atual for maior que o permitido, recorta os últimos `context_size` tokens
        # Isso evita erro de input com mais tokens que o modelo pode processar
        idx_cond = idx[:, -context_size:]

        # Desliga o cálculo de gradiente, pois estamos em modo de inferência
        with torch.no_grad():
            # Executa o modelo sobre o contexto para obter os logits (pontuações brutas dos tokens)
            logits = model(idx_cond)

        # Pegamos apenas a predição do último token gerado (último passo de tempo)
        # logits tem shape (batch, seq_len, vocab_size), então selecionamos `-1` no eixo seq_len
        logits = logits[:, -1, :]

        # Seleciona o token mais provável com base nos logits (argmax ao longo do vocabulário)
        # Isso retorna o índice do token mais provável, com shape (batch, 1)
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Anexa esse token gerado à sequência existente de entrada
        # Isso permite ao modelo continuar a geração no próximo passo
        # Novo shape: (batch, tokens + 1)
        idx = torch.cat((idx, idx_next), dim=1)

    # Retorna toda a sequência com os novos tokens gerados anexados
    return idx


def generate_temperature(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Gera uma sequência de texto a partir de um modelo de linguagem, com suporte a amostragem controlada
    por temperatura, top-k sampling e parada antecipada via token de fim de sequência (EOS).

    Parâmetros:
    -----------
    model : torch.nn.Module
        Modelo autoregressivo de linguagem treinado (como um Transformer).

    idx : torch.Tensor
        Tensor contendo a sequência de entrada (batch_size, seq_len) com os tokens já conhecidos.

    max_new_tokens : int
        Número máximo de novos tokens a serem gerados pela função.

    context_size : int
        Número máximo de tokens que o modelo consegue processar como entrada (janela de contexto).

    temperature : float, opcional (default = 0.0)
        Fator que controla a aleatoriedade da geração:
        - Se 0.0, usa `argmax` (decodificação determinística).
        - Se > 0.0, aplica softmax e sampling proporcional aos logits.

    top_k : int, opcional
        Se especificado, restringe a distribuição de saída aos `k` tokens mais prováveis (top-k sampling).

    eos_id : int, opcional
        ID do token de fim de sequência. Se for gerado, a geração para imediatamente.

    Retorna:
    --------
    torch.Tensor
        Tensor com a sequência original estendida com os novos tokens gerados.
    """

    # Itera para gerar até max_new_tokens
    for _ in range(max_new_tokens):

        # Recorta os últimos tokens até o tamanho máximo do contexto permitido
        idx_cond = idx[:, -context_size:]

        # Desativa gradientes para evitar computações desnecessárias durante a inferência
        with torch.no_grad():
            logits = model(idx_cond)  # Obtém as predições do modelo

        # Foca apenas na última predição de token (última posição na sequência)
        logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # Se o top_k estiver definido, aplica o filtro para manter apenas os k maiores logits
        if top_k is not None:
            # Obtém os top_k maiores logits
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # Valor mínimo entre os top_k
            # Define os logits fora do top_k como -inf, para que tenham probabilidade ~0
            logits = torch.where(logits < min_val, torch.tensor(
                float("-inf")).to(logits.device), logits)

        # Se a temperatura for maior que 0, aplica amostragem probabilística
        if temperature > 0.0:
            logits = logits / temperature  # Escala os logits com a temperatura

            # Converte os logits em probabilidades com softmax
            probs = torch.softmax(logits, dim=-1)  # (batch_size, vocab_size)

            # Amostra aleatoriamente o próximo token com base nas probabilidades
            idx_next = torch.multinomial(
                probs, num_samples=1)  # (batch_size, 1)

        # Caso contrário, usa a predição determinística com maior logit (greedy decoding)
        else:
            idx_next = torch.argmax(
                logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # Se um token de fim de sequência for gerado, interrompe a geração antecipadamente
        if eos_id is not None and (idx_next == eos_id).all():
            break

        # Adiciona o novo token gerado à sequência existente
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, seq_len+1)

    # Retorna a sequência completa com os novos tokens adicionados
    return idx


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """
    Cria um DataLoader PyTorch a partir de um texto bruto, usando tokenização baseada no modelo GPT-2
    e segmentação com sobreposição para treinar modelos de linguagem autoregressivos.

    Parâmetros:
    -----------
    txt : str
        Texto de entrada bruto que será tokenizado e segmentado em exemplos de treinamento.

    batch_size : int, opcional (default = 4)
        Número de amostras por lote retornadas pelo DataLoader.

    max_length : int, opcional (default = 256)
        Tamanho máximo (em tokens) de cada sequência de entrada usada no treinamento.

    stride : int, opcional (default = 128)
        Número de tokens para avançar entre janelas de texto. Se for menor que max_length,
        as janelas terão sobreposição.

    shuffle : bool, opcional (default = True)
        Define se os dados devem ser embaralhados a cada época.

    drop_last : bool, opcional (default = True)
        Se True, descarta o último lote caso ele tenha menos amostras que `batch_size`.

    num_workers : int, opcional (default = 0)
        Número de subprocessos a serem usados para carregar os dados. 0 significa que a
        carga será feita no processo principal.

    Retorna:
    --------
    torch.utils.data.DataLoader
        DataLoader configurado para iterar sobre os dados tokenizados e segmentados, pronto para uso
        no treinamento ou avaliação de modelos de linguagem.
    """
    # Inicializa o tokenizador com base no modelo GPT-2
    # O tokenizador converte texto em tokens numéricos compatíveis com modelos do tipo GPT
    tokenizer = tiktoken.get_encoding("gpt2")

    # Cria uma instância do dataset personalizado GPTDatasetV1
    # Esse dataset deve cuidar da tokenização, segmentação em blocos e aplicação de stride (sobreposição)
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Cria o DataLoader do PyTorch para alimentar o modelo durante o treinamento ou inferência
    # - batch_size: número de amostras por lote
    # - shuffle: se True, embaralha os dados a cada época
    # - drop_last: descarta o último lote se ele tiver menos que batch_size
    # - num_workers: número de subprocessos para carregar os dados (0 = carregar no processo principal)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    # Retorna o dataloader pronto para uso no loop de treinamento
    return dataloader


def token_ids_to_text(token_ids, tokenizer):
    """
    Converte uma sequência de IDs de tokens em texto legível.

    Parâmetros:
    -----------
    token_ids : torch.Tensor
        Tensor contendo os IDs dos tokens, geralmente no formato (1, seq_len) para representar
        uma única sequência com dimensão de batch.

    tokenizer : Tokenizer
        Um tokenizador compatível (por exemplo, do Hugging Face ou Tiktoken) usado para converter
        IDs de volta em texto.

    Retorna:
    --------
    str
        Texto decodificado a partir da sequência de tokens fornecida.
    """

    # Remove a dimensão de batch (por exemplo, de shape (1, seq_len) para (seq_len,))
    flat = token_ids.squeeze(0)

    # Converte a sequência de IDs em uma lista e a decodifica para string
    return tokenizer.decode(flat.tolist())


def text_to_token_ids(text, tokenizer):
    """
    Converte um texto em IDs de tokens utilizando o tokenizador fornecido.

    Parâmetros:
    -----------
    text : str
        Texto de entrada a ser tokenizado.

    tokenizer : Tokenizer
        Um tokenizador compatível (ex: Hugging Face, Tiktoken) com método .encode().

    Retorna:
    --------
    torch.Tensor
        Tensor com IDs dos tokens no formato (1, seq_len), incluindo a dimensão de batch.
    """
    # Codifica o texto, permitindo o token especial '<|endoftext|>'
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})

    # Converte para tensor e adiciona uma dimensão de batch (1, seq_len)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensor


def create_balanced_dataset(df):
    """
    Cria um DataFrame balanceado com o mesmo número de instâncias 'spam' e 'ham'.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame original contendo uma coluna 'Label' com valores 'spam' ou 'ham'.

    Retorna:
    --------
    pandas.DataFrame
        DataFrame balanceado com igual número de exemplos para cada classe.
    """
    # Conta o número de instâncias com o rótulo 'spam'
    num_spam = df[df["Label"] == "spam"].shape[0]

    # Amostra aleatoriamente instâncias 'ham' para igualar a quantidade de 'spam'
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # Combina o subconjunto de 'ham' com todas as instâncias 'spam'
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df


def random_split(df, train_frac, validation_frac):
    """
    Divide um DataFrame em conjuntos de treino, validação e teste com base em proporções fornecidas.

    Parâmetros:
    -----------
    df : pandas.DataFrame
        DataFrame completo a ser dividido.

    train_frac : float
        Fração dos dados a ser usada para treino (ex: 0.7 para 70%).

    validation_frac : float
        Fração dos dados a ser usada para validação (ex: 0.15 para 15%).

    Retorna:
    --------
    tuple:
        (train_df, validation_df, test_df) — três subconjuntos do DataFrame original.
    """
    # Embaralha o DataFrame para garantir aleatoriedade
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # Calcula os índices de divisão
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # Cria os subconjuntos
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Gera uma sequência de tokens a partir de uma entrada inicial (`idx`) usando um modelo autoregressivo.

    Parâmetros:
    - model: modelo de linguagem (ex: Transformer) que recebe tokens e retorna logits.
    - idx (torch.Tensor): tensor de entrada de tokens com shape (batch_size, seq_len).
    - max_new_tokens (int): número máximo de novos tokens a serem gerados.
    - context_size (int): número de tokens do contexto usado na entrada do modelo.
    - temperature (float): fator de aleatoriedade para amostragem (0.0 = determinístico).
    - top_k (int | None): se fornecido, aplica top-k sampling (limita a amostragem aos `k` tokens mais prováveis).
    - eos_id (int | None): se fornecido, para a geração se o token `eos_id` for gerado.

    Retorna:
    - idx (torch.Tensor): sequência original expandida com até `max_new_tokens` tokens gerados.
    """

    # Laço principal de geração: um novo token por iteração, até atingir max_new_tokens
    for _ in range(max_new_tokens):
        # Seleciona apenas os últimos `context_size` tokens para alimentar o modelo
        idx_cond = idx[:, -context_size:]

        # Desativa o cálculo do gradiente para economizar memória e acelerar a inferência
        with torch.no_grad():
            # Output shape: (batch_size, seq_len, vocab_size)
            logits = model(idx_cond)

        # Pegamos apenas os logits do último token gerado
        logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

        # Aplica top-k sampling se top_k for fornecido
        if top_k is not None:
            # Obtém os `top_k` logits mais altos e seus valores
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # menor valor entre os top_k

            # Zera (coloca -inf) todos os logits que não estão entre os top_k
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )

        # Aplica a temperatura, se maior que zero
        if temperature > 0.0:
            logits = logits / temperature  # escala os logits

            # Converte logits em probabilidades com softmax
            probs = torch.softmax(logits, dim=-1)

            # Amostra um token da distribuição (estocástico)
            idx_next = torch.multinomial(
                probs, num_samples=1)  # Shape: (batch_size, 1)

        else:
            # Se temperatura = 0, escolhe o token com maior probabilidade (determinístico)
            # Shape: (batch_size, 1)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # Se um token de fim de sequência (`eos_id`) for gerado, interrompe a geração
        if eos_id is not None and torch.all(idx_next == eos_id):
            break

        # Anexa o novo token gerado à sequência existente
        # Shape: (batch_size, seq_len + 1)
        idx = torch.cat((idx, idx_next), dim=1)

    # Retorna a sequência completa (original + tokens gerados)
    return idx


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    """
    Função personalizada para preparação (collate) de lotes de dados para treinamento de modelos de linguagem.

    Esta função realiza:
    - Padding das sequências no lote até o tamanho da maior sequência + token de término.
    - Criação de pares (input, target) deslocados.
    - Máscara para ignorar perdas sobre os tokens de padding.
    - Truncamento opcional para um tamanho máximo de sequência.

    Parâmetros:
    - batch (List[List[int]]): Lista de sequências de tokens (listas de inteiros).
    - pad_token_id (int): ID do token de padding (padrão: 50256 para <|endoftext|> do GPT-2).
    - ignore_index (int): Índice a ser ignorado na função de perda (usado para ignorar o padding).
    - allowed_max_length (int | None): Tamanho máximo permitido para a sequência (input e target).
    - device (str): Dispositivo de destino para os tensores (ex: "cuda" ou "cpu").

    Retorna:
    - inputs_tensor (torch.Tensor): Tensor dos inputs, shape (batch_size, seq_len).
    - targets_tensor (torch.Tensor): Tensor dos targets (deslocados), shape (batch_size, seq_len).
    """

    # Descobre o tamanho máximo da sequência no lote (adicionando 1 para o token <|endoftext|>)
    batch_max_length = max(len(item) + 1 for item in batch)

    # Listas para armazenar os tensores resultantes
    inputs_lst, targets_lst = [], []

    # Itera sobre cada item do lote
    for item in batch:
        new_item = item.copy()  # Evita modificar a lista original

        # Adiciona o token <|endoftext|> no final
        new_item += [pad_token_id]

        # Aplica padding até que todos tenham o mesmo tamanho
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))

        # Cria input truncando o último token
        inputs = torch.tensor(padded[:-1])  # (t0, ..., tn)
        # Cria target deslocando para a direita
        targets = torch.tensor(padded[1:])  # (t1, ..., tn+1)

        # Cria uma máscara para encontrar os índices de padding nos targets
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()

        # Substitui todos os tokens de padding (exceto o primeiro) por `ignore_index` para ignorar na perda
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # Opcionalmente corta a sequência se `allowed_max_length` for especificado
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        # Adiciona os tensores às listas
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Empilha as listas e transfere para o dispositivo especificado (CPU ou GPU)
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


def generate5(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Gera uma sequência de texto usando um modelo LLM autoregressivo (como GPT).

    Parâmetros:
    -----------
    model : torch.nn.Module
        Modelo de linguagem treinado, que recebe índices de tokens e retorna logits.
    idx : torch.Tensor
        Tensor de entrada contendo os tokens iniciais (shape: [batch_size, seq_len]).
    max_new_tokens : int
        Número máximo de novos tokens a serem gerados.
    context_size : int
        Tamanho do contexto a ser considerado (limite da janela de entrada).
    temperature : float, opcional
        Escala aplicada aos logits antes da amostragem (valor padrão: 0.0, sem aleatoriedade).
    top_k : int, opcional
        Se definido, restringe a amostragem apenas ao top_k tokens com maior probabilidade.
    eos_id : int, opcional
        ID do token de fim de sequência. Se encontrado, a geração para antecipadamente.

    Retorna:
    --------
    idx : torch.Tensor
        Sequência completa (tokens iniciais + tokens gerados).
    """
    for _ in range(max_new_tokens):
        # Garante que só os últimos 'context_size' tokens sejam usados como entrada
        idx_cond = idx[:, -context_size:]

        # Desativa o cálculo de gradiente para economia de memória e velocidade
        with torch.no_grad():
            # shape: (batch_size, seq_len, vocab_size)
            logits = model(idx_cond)

        # Mantém apenas os logits do último token gerado
        logits = logits[:, -1, :]  # shape: (batch_size, vocab_size)

        if top_k is not None:
            # Obtém os top_k maiores valores de logit
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]  # Valor mínimo entre os top_k

            # Substitui logits menores que o mínimo com -inf (remoção da probabilidade)
            logits = torch.where(logits < min_val,
                                 torch.tensor(float("-inf")).to(logits.device),
                                 logits)

        if temperature > 0.0:
            # Escala os logits para controlar a aleatoriedade da geração
            logits = logits / temperature

            # Converte logits em probabilidades com softmax
            probs = torch.softmax(logits, dim=-1)

            # Amostra o próximo token com base na distribuição de probabilidade
            idx_next = torch.multinomial(
                probs, num_samples=1)  # shape: (batch_size, 1)

        else:
            # Seleciona o token com maior probabilidade
            # shape: (batch_size, 1)
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if eos_id is not None and torch.any(idx_next == eos_id):
            break  # Interrompe a geração ao encontrar o token de fim

        # shape: (batch_size, seq_len + 1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
