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
