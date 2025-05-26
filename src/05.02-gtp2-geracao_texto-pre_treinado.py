import torch
import tiktoken

from models.gpt_model import GPTModel
from function.geral import (
    text_to_token_ids, token_ids_to_text, generate5
)

torch.manual_seed(123)


##################################################################################
# Parte 1 - Configuração do modelo GPT-2 (small)
##################################################################################

GPT_CONFIG_124M = {
    "vocab_size": 50257,        # Tamanho do vocabulário utilizado pelo tokenizer
    # Tamanho máximo da janela de contexto (tokens por entrada); reduzido para 256 (original é 1024)
    "context_length": 256,
    # Dimensão dos vetores de embedding (tamanho da representação de cada token)
    "emb_dim": 768,
    # Número de cabeças de atenção no mecanismo de atenção multi-head
    "n_heads": 12,
    "n_layers": 12,             # Número de camadas do transformador
    # Taxa de dropout, usada para evitar overfitting durante o treinamento
    "drop_rate": 0.1,
    # Se deve usar viés (bias) nos cálculos de Query, Key e Value
    "qkv_bias": False
}

##################################################################################
# Parte 2 - Configuração do dispositivo (CPU ou GPU) e inicialização do modelo
##################################################################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Detecta se há GPU disponível; caso contrário, usa CPU

model = GPTModel(GPT_CONFIG_124M)
# Cria uma instância do modelo GPT-2 com a configuração definida acima

model.to(device)
# Move o modelo para o dispositivo (GPU ou CPU)

##################################################################################
# Parte 3 - Otimizador (usado para treinamento, mesmo que não seja utilizado aqui)
##################################################################################

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
# Inicializa o otimizador AdamW com taxa de aprendizado e fator de regularização (weight decay)

##################################################################################
# Parte 4 - Tokenizador e nova geração de texto com parâmetros personalizados
##################################################################################

tokenizer = tiktoken.get_encoding("gpt2")
# Carrega o tokenizador compatível com o GPT-2

model = GPTModel(GPT_CONFIG_124M)
# Cria novamente o modelo (esta linha sobrescreve o modelo anterior)

token_ids = generate5(
    model=model,
    # Converte o texto inicial em IDs de tokens
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,             # Número máximo de novos tokens a serem gerados
    # Define o tamanho do contexto para geração
    context_size=GPT_CONFIG_124M["context_length"],
    # Usa Top-k sampling (considera os 25 tokens mais prováveis em cada passo)
    top_k=25,
    # Controla a aleatoriedade na geração (maior = mais criativo/aleatório)
    temperature=1.4
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
# Converte os tokens gerados de volta para texto e imprime
