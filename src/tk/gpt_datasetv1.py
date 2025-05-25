import tiktoken
import torch
from torch.utils.data import Dataset


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # Listas para armazenar os pares de entrada e alvo
        self.input_ids = []
        self.target_ids = []

        # Tokeniza todo o texto, incluindo tokens especiais como "<|endoftext|>"
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Usa uma janela deslizante para criar sequências de entrada e saída com sobreposição
        for i in range(0, len(token_ids) - max_length, stride):
            # Sequência de entrada com tamanho fixo
            input_chunk = token_ids[i:i + max_length]
            # Sequência alvo: deslocada uma posição à frente
            target_chunk = token_ids[i + 1: i + max_length + 1]
            # Armazena como tensores
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # Retorna o número de pares (input, target)
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Retorna o par correspondente ao índice solicitado
        return self.input_ids[idx], self.target_ids[idx]
