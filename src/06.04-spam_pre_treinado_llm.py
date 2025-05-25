import torch
import tiktoken
import pandas as pd
from torch.utils.data import DataLoader

from spam.spam_dataset import SpamDataset
from spam.spam_functions import classify_review
from download.gpt_download import download_and_load_gpt2
from layer.layer_function import load_weights_into_gpt
from models.gpt_model import GPTModel

torch.manual_seed(123)

################################################
# Parte 1 - Carregando e configurando o modelo
################################################

# Define o modelo GPT-2 que será utilizado entre as opções disponíveis
CHOOSE_MODEL = "gpt2-medium (355M)"

# Prompt de entrada (inicial) para alimentar o modelo
INPUT_PROMPT = "Every effort moves"

# Configurações base comuns a todos os modelos GPT-2
BASE_CONFIG = {
    "vocab_size": 50257,     # Tamanho do vocabulário (padrão do GPT-2)
    "context_length": 1024,  # Tamanho máximo da sequência de entrada (tokens)
    "drop_rate": 0.0,        # Taxa de dropout (0.0 desativa o dropout)
    # Se True, adiciona bias às projeções de query, key e value na atenção
    "qkv_bias": True
}

# Dicionário com configurações específicas para cada tamanho de modelo GPT-2
model_configs = {
    # GPT-2 pequeno: 124 milhões de parâmetros, 12 camadas, 12 cabeças, dimensão do embedding 768
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},

    # GPT-2 médio: 355 milhões de parâmetros, 24 camadas, 16 cabeças, dimensão do embedding 1024
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},

    # GPT-2 grande: 774 milhões de parâmetros, 36 camadas, 20 cabeças, dimensão do embedding 1280
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},

    # GPT-2 XL: 1.558 bilhões de parâmetros, 48 camadas, 25 cabeças, dimensão do embedding 1600
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


# Atualiza o dicionário BASE_CONFIG com os parâmetros específicos do modelo escolhido (sobrescreve/mescla)
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Cria uma instância do modelo GPTModel com a configuração completa
model = GPTModel(BASE_CONFIG)

# Extrai o número de parâmetros do modelo da string, ex: "355M" de "gpt2-medium (355M)"
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

# Faz o download e carrega os pesos originais do GPT-2 (retorna configurações e pesos)
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2")

# Carrega os pesos do modelo GPT-2 no modelo recém-criado
load_weights_into_gpt(model, params)

# Coloca o modelo em modo de avaliação (desativa dropout e batch norm, se houver)
model.eval()

# Inicializa o tokenizer compatível com o GPT-2
tokenizer = tiktoken.get_encoding("gpt2")

# Define o dispositivo de execução (GPU se disponível, senão CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move o modelo para o dispositivo selecionado (GPU ou CPU)
model.to(device)

# Observação: não é necessário fazer `model = model.to(device)` explicitamente,
# pois `nn.Module.to()` modifica o objeto in-place e retorna a mesma instância.

################################################
# Parte 2 - Carregando o conjunto de dados
################################################

# Cria o dataset de treino a partir do arquivo CSV com a classe SpamDataset
# max_length=None significa que o comprimento máximo das sequências será determinado automaticamente
train_dataset = SpamDataset(
    csv_file="data/spam/train.csv",
    max_length=None,
    tokenizer=tokenizer
)

# Cria o dataset de validação usando o arquivo CSV correspondente
# Utiliza o mesmo max_length do dataset de treino para consistência
val_dataset = SpamDataset(
    csv_file="data/spam/validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

# Cria o dataset de teste também com o mesmo max_length do dataset de treino
test_dataset = SpamDataset(
    csv_file="data/spam/test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

# Número de processos paralelos para carregar dados (0 = carregamento no processo principal)
num_workers = 8

# Tamanho do batch para o treinamento e validação
batch_size = 8

# DataLoader para o dataset de treino, com shuffle para embaralhar os dados a cada época
# drop_last=True para descartar o último batch se ele estiver incompleto (útil para batch normalization)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

# DataLoader para validação, sem embaralhamento e não descartando o último batch incompleto
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

# DataLoader para teste, com as mesmas configurações do val_loader
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)


################################################
# Parte 3 - Fazendo a inferência
################################################


text_1 = (
    "You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award."
)

print(classify_review(
    text_1, model, tokenizer, device, max_length=train_dataset.max_length
))


text_2 = (
    "Hey, just wanted to check if we're still on"
    " for dinner tonight? Let me know!"
)

print(classify_review(
    text_2, model, tokenizer, device, max_length=train_dataset.max_length
))

################################################
# Parte 4 - Salvando e carregando o modelo
################################################

# torch.save(model.state_dict(), "review_classifier.pth")
# model_state_dict = torch.load("review_classifier.pth", map_location=device, weights_only=True)
# model.load_state_dict(model_state_dict)
