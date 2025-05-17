# Importa o tokenizador tiktoken (usado com modelos como GPT)
import tiktoken
# Importa o PyTorch
import torch
# Importa as classes para criação de dataset e dataloader
from torch.utils.data import Dataset, DataLoader


# Define uma classe de dataset personalizada para treinar modelos autoregressivos (como o GPT)
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


# Função auxiliar para criar o DataLoader com base no texto bruto
def create_dataloader_v1(txt, batch_size, max_length, stride,
                         shuffle=True, drop_last=True, num_workers=0):
    # Inicializa o tokenizador GPT-2
    tokenizer = tiktoken.get_encoding("gpt2")

    # Cria o dataset com base no texto e nos parâmetros
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Cria o DataLoader a partir do dataset
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


# Abre e lê o conteúdo do arquivo de texto
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# Define o tamanho do vocabulário (GPT-2 tem 50257 tokens)
vocab_size = 50257
# Dimensão dos vetores de embedding (número de features por token)
output_dim = 256
# Comprimento máximo do contexto (quantos tokens considerar ao mesmo tempo)
context_length = 1024

# Camada de embedding para os tokens
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# Camada de embedding para as posições (posição 0 até context_length - 1)
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

# Define os parâmetros para o DataLoader
batch_size = 8
max_length = 4  # Tamanho de sequência (curto neste exemplo para demonstração)
# Cria o DataLoader com as sequências processadas
dataloader = create_dataloader_v1(
    raw_text,
    batch_size=batch_size,
    max_length=max_length,
    stride=max_length  # Sem sobreposição (stride = max_length)
)

# Itera sobre os batches do DataLoader (pegando apenas o primeiro)
for batch in dataloader:
    x, y = batch  # x = sequência de entrada, y = sequência alvo

    # Obtém os embeddings dos tokens (shape: [batch_size, max_length, output_dim])
    token_embeddings = token_embedding_layer(x)
    # Cria a sequência de posições (0 até max_length - 1) e obtém seus embeddings
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))

    # Soma os embeddings de token e posição (broadcasting nas posições)
    input_embeddings = token_embeddings + pos_embeddings

    # Interrompe o loop após o primeiro batch
    break

# Mostra o formato do tensor final de embeddings (esperado: [batch_size, max_length, output_dim])
print()
print(input_embeddings.shape)
print()
