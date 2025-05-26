import torch
import tiktoken

from models.gpt_model import GPTModel
from function.geral import (
    text_to_token_ids,
    token_ids_to_text,
    generate_text_simple,
    create_dataloader_v1,
    generate5
)

from weights.weights import load_weights_into_gpt
from download.gpt_download import download_and_load_gpt2


##################################################################################
# Parte 1 - Define a configuração do modelo GPT-2 pequeno (124M)
##################################################################################

GPT_CONFIG_124M = {
    "vocab_size": 50257,        # Tamanho do vocabulário do modelo GPT-2
    # Tamanho do contexto (quantidade de tokens de entrada)
    "context_length": 256,
    "emb_dim": 768,             # Dimensão dos embeddings
    "n_heads": 12,              # Número de cabeças de atenção
    "n_layers": 12,             # Número de camadas Transformer
    "drop_rate": 0.1,           # Taxa de dropout
    "qkv_bias": False           # Se usa viés nos pesos QKV
}

##################################################################################
# Parte 2 - Cria modelo e gera texto simples com prompt inicial
##################################################################################

# Define semente aleatória para reprodutibilidade
torch.manual_seed(123)
# Instancia o modelo com a configuração definida
model = GPTModel(GPT_CONFIG_124M)

start_context = "Every effort moves you"    # Texto de início (prompt)
tokenizer = tiktoken.get_encoding("gpt2")  # Instancia o tokenizador GPT-2

token_ids = generate_text_simple(           # Gera texto com modelo (sem treinamento)
    model=model,
    # Converte prompt para tokens
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,                      # Gera no máximo 10 novos tokens
    context_size=GPT_CONFIG_124M["context_length"]     # Tamanho do contexto
)
print("\nOutput text:\n", token_ids_to_text(
    token_ids, tokenizer))  # Exibe texto gerado
print()


##################################################################################
# Parte 3 - Leitura de arquivo de texto para treinamento/validação
##################################################################################

# Caminho do arquivo de texto de entrada
file_path = "data/the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()                 # Lê todo conteúdo do arquivo


##################################################################################
# Parte 4 - Divide dados em treino e validação
##################################################################################

# Percentual dos dados para treino
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))          # Índice de corte
train_data = text_data[:split_idx]                     # Dados de treino
val_data = text_data[split_idx:]                       # Dados de validação

##################################################################################
# Parte 5 - Cria DataLoaders para treino e validação
##################################################################################

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

##################################################################################
# Parte 8 - Detecta se há GPU disponível
##################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##################################################################################
# Parte 9 - Define configurações alternativas para modelos maiores
##################################################################################

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

model_name = "gpt2-small (124M)"             # Escolhe modelo
NEW_CONFIG = GPT_CONFIG_124M.copy()          # Copia configuração base
# Atualiza com parâmetros específicos do modelo
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True}
                  )  # Ajusta contexto e viés


##################################################################################
# Parte 10 - Instancia modelo com nova configuração e carrega pesos
##################################################################################

gpt = GPTModel(NEW_CONFIG)      # Cria novo modelo com a configuração ajustada
gpt.eval()                      # Coloca o modelo em modo de avaliação

settings, params = download_and_load_gpt2(   # Baixa pesos do modelo GPT-2 124M
    model_size="124M",
    models_dir="gpt2"
)
load_weights_into_gpt(gpt, params)           # Carrega pesos baixados no modelo
# Move modelo para GPU (se disponível)
gpt.to(device)


##################################################################################
# Parte 11 - Loop principal para geração de tokens com parâmetros customizados
##################################################################################

token_ids = generate5(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(
        device),  # Prompt tokenizado
    max_new_tokens=25,              # Gera até 25 novos tokens
    context_size=NEW_CONFIG["context_length"],  # Tamanho do contexto
    # Aplica filtro top-k (amostragem dos 50 melhores)
    top_k=50,
    temperature=1.5                 # Controla a aleatoriedade da geração
)

print("Output text:\n", token_ids_to_text(
    token_ids, tokenizer))  # Exibe o texto gerado
