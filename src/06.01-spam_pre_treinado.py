import tiktoken
import pandas as pd

from models.gpt_model import GPTModel
from spam.spam_dataset import SpamDataset
from weights.weights import load_weights_into_gpt
from download.gpt_download import download_and_load_gpt2
from function.geral import create_balanced_dataset, random_split, generate_text_simple, text_to_token_ids, token_ids_to_text


################################################################################################
# Parte 1 - Carregando e preparando o conjunto de dados
################################################################################################


df = pd.read_csv('data/sms_spam_collection/SMSSpamCollection.tsv',
                 sep="\t", header=None, names=["Label", "Text"])

balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# Inicializa o tokenizer compatível com o GPT-2
tokenizer = tiktoken.get_encoding("gpt2")

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


################################################
# Parte 2 - Carregando e configurando o modelo
################################################

# Define o modelo GPT-2 que será utilizado entre as opções disponíveis
CHOOSE_MODEL = "gpt2-medium (124M)"

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

# Extrai o número de parâmetros do modelo da string, ex: "1424M" de "gpt2-small (124M)"
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# Faz o download e carrega os pesos originais do GPT-2 (retorna configurações e pesos)
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2")

# Cria uma instância do modelo GPTModel com a configuração completa
model = GPTModel(BASE_CONFIG)
# Carrega os pesos do modelo GPT-2 no modelo recém-criado
load_weights_into_gpt(model, params)
# Coloca o modelo em modo de avaliação (desativa dropout e batch norm, se houver)
model.eval()

################################################
# Parte 3 - Fazendo a inferência
################################################

# Define a string inicial para geração de texto
text_1 = "Every effort moves you"

# Converte o texto inicial em IDs de tokens e gera até 15 novos tokens usando o modelo
token_ids = generate_text_simple(
    model=model,  # modelo de linguagem utilizado para gerar texto
    # converte o texto para tensor de token IDs
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,  # número máximo de tokens a serem gerados
    # tamanho máximo do contexto usado pelo modelo
    context_size=BASE_CONFIG["context_length"]
)

# Converte os token IDs gerados de volta para texto e imprime o resultado
print(token_ids_to_text(token_ids, tokenizer))

# Define um segundo texto com uma pergunta para o modelo sobre spam
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    " 'You are a winner you have been specially"
    " selected to receive $1000 cash or a $2000 award.'"
)

# Converte o segundo texto em IDs de tokens e gera até 23 novos tokens com o modelo
token_ids = generate_text_simple(
    model=model,  # mesmo modelo de linguagem
    idx=text_to_token_ids(text_2, tokenizer),  # converte o texto em token IDs
    max_new_tokens=23,  # número máximo de tokens a gerar para essa entrada
    # tamanho máximo do contexto para a geração
    context_size=BASE_CONFIG["context_length"]
)

# Converte os token IDs gerados para texto e imprime o resultado final
print(token_ids_to_text(token_ids, tokenizer))
