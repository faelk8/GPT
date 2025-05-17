import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.target_ids = []

        # Tokeniza o texto completo, incluindo tokens especiais permitidos
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # Cria janelas deslizantes com sobreposição usando stride
        # Isso gera pares de (input, target) com deslocamento de 1 token
        for i in range(0, len(token_ids) - max_length, stride):
            # Entrada do modelo
            input_chunk = token_ids[i:i + max_length]
            # Alvo, deslocado 1 token à frente
            target_chunk = token_ids[i + 1: i + max_length + 1]

            # Armazena como tensores para uso com PyTorch
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # Retorna o número de amostras criadas (quantas janelas foram extraídas)
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Retorna o par (entrada, alvo) no índice `idx`
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Inicializa o tokenizer do GPT-2 usando a biblioteca tiktoken (rápida e precisa)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Cria o dataset a partir do texto e parâmetros
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Cria o DataLoader com suporte a batching, embaralhamento e paralelismo
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,     # Remove o último batch se ele for menor que o batch_size
        num_workers=num_workers  # Número de subprocessos usados na leitura dos dados
    )

    return dataloader
