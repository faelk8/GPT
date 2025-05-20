import torch
import tiktoken
import os
import urllib.request
from torch.utils.data import Dataset, DataLoader

from src.token.gptdatasetv1 import GPTDatasetV1


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
    # idx é um tensor de shape (batch, n_tokens), contendo os índices dos tokens já gerados
    # A função vai gerar até max_new_tokens novos tokens por amostra no batch

    for _ in range(max_new_tokens):

        # Se a sequência de entrada for maior que o tamanho máximo de contexto,
        # cortamos apenas os últimos context_size tokens (janela deslizante)
        idx_cond = idx[:, -context_size:]

        # Realiza inferência sem cálculo de gradientes (mais eficiente para geração)
        with torch.no_grad():
            logits = model(idx_cond)  # Saída: [batch, seq_len, vocab_size]

        # Pegamos apenas os logits do último token gerado
        logits = logits[:, -1, :]  # Reduz para [batch, vocab_size]

        # Convertemos os logits em probabilidades com softmax
        probas = torch.softmax(logits, dim=-1)  # [batch, vocab_size]

        # Escolhemos o índice do token com maior probabilidade (greedy decoding)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # [batch, 1]

        # Concatenamos esse novo token à sequência existente
        idx = torch.cat((idx, idx_next), dim=1)  # [batch, n_tokens+1]

    # Retorna a sequência original com os novos tokens adicionados
    return idx


def generate_temperature(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """
    Gera novos tokens usando um modelo de linguagem com suporte a temperatura, top-k sampling e token de parada.

    Parâmetros:
    - model: o modelo de linguagem treinado (por exemplo, um Transformer)
    - idx: tensor inicial de índices de tokens (batch_size, seq_len)
    - max_new_tokens: número máximo de novos tokens a serem gerados
    - context_size: número máximo de tokens de contexto que o modelo pode usar
    - temperature: fator de aleatoriedade aplicado aos logits
    - top_k: se especificado, restringe a amostragem aos top-k tokens mais prováveis
    - eos_id: se especificado, para a geração se esse token for encontrado
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
