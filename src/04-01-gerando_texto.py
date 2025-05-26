import tiktoken
import torch
from models.gpt_model import GPTModel
from layer.layer_norm import LayerNorm
from layer.feed_forward import FeedForward
from transformer.transformer_block import TransformerBlock
from activation.gelu import GELU
from function.geral import generate_text_simple

torch.manual_seed(123)

##################################################################################
# Parte 1 - Inicialização do tokenizer e preparação do batch de entrada
##################################################################################
# Inicializa o tokenizer com o vocabulário do GPT-2
tokenizer = tiktoken.get_encoding("gpt2")
batch = []  # Lista vazia para armazenar tensores codificados das frases

txt1 = "Every effort moves you"  # Primeira frase de exemplo
txt2 = "Every day holds a"        # Segunda frase de exemplo

# Codifica a primeira frase em tokens e adiciona à lista
batch.append(torch.tensor(tokenizer.encode(txt1)))
# Codifica a segunda frase em tokens e adiciona à lista
batch.append(torch.tensor(tokenizer.encode(txt2)))
# Empilha os tensores em uma dimensão para criar um batch (tensor 2D)
batch = torch.stack(batch, dim=0)

##################################################################################
# Parte 2 - Definição da configuração do modelo GPT-2
##################################################################################
GPT_CONFIG_124M = {
    "vocab_size": 50257,    # Tamanho do vocabulário do GPT-2 padrão
    # Tamanho máximo da sequência de entrada (context window)
    "context_length": 1024,
    # Dimensão do embedding (representação interna das palavras)
    "emb_dim": 768,
    "n_heads": 12,          # Número de cabeças de atenção no mecanismo de self-attention
    "n_layers": 12,         # Número de camadas do Transformer
    "drop_rate": 0.1,       # Taxa de dropout para regularização durante o treinamento
    "qkv_bias": False       # Se deve usar bias nos cálculos das queries, keys e values
}

##################################################################################
# Parte 3 - Criação do modelo GPT-2 e execução do forward pass
##################################################################################
# Instancia o modelo GPT com a configuração definida
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)  # Passa o batch pelo modelo para obter a saída (logits)
print("Input batch:\n", batch)   # Exibe o batch de entrada (tokens)
# Exibe o formato da saída do modelo (batch_size, seq_len, vocab_size)
print("\nOutput shape:", out.shape)
print(out)                        # Exibe os valores da saída do modelo

##################################################################################
# Parte 4 - Preparação do texto de contexto para geração
##################################################################################
start_context = "Hello, I am"           # Texto inicial para a geração do modelo
# Codifica o texto de contexto em tokens
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)               # Exibe a lista de tokens codificados

# Converte para tensor e adiciona dimensão de batch (1, seq_len)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# Exibe a forma do tensor de entrada
print("encoded_tensor.shape:", encoded_tensor.shape)

##################################################################################
# Parte 5 - Geração de texto com o modelo em modo avaliação
##################################################################################
model.eval()  # Coloca o modelo em modo avaliação (desliga dropout e batchnorm)

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,                 # Entrada codificada para iniciar a geração
    max_new_tokens=6,                   # Número máximo de tokens a serem gerados
    # Context window usada na geração
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)                 # Exibe os tokens gerados pelo modelo
print("Output length:", len(out[0]))  # Exibe o comprimento da sequência gerada

# Decodifica os tokens gerados para texto legível
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)                   # Exibe o texto gerado pelo modelo
