import time
import pandas as pd
import torch
from torch.utils.data import DataLoader

import tiktoken

from download.gpt_download import download_and_load_gpt2
from spam.spam_dataset import SpamDataset
from models.gpt_model import GPTModel
from weights.weights import load_weights_into_gpt
from function.geral import create_balanced_dataset, random_split
from metrics.metrics import calc_accuracy_loader, calc_loss_loader
from plots.plots import plot_values
from train.train_model import train_classifier_simple


# Configurações iniciais
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################################
# Parte 1 - Carregando e preparando o conjunto de dados
################################################################################################

df = pd.read_csv('data/sms_spam_collection/SMSSpamCollection.tsv',
                 sep="\t", header=None, names=["Label", "Text"])
balanced_df = create_balanced_dataset(df)
balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

train_df, val_df, test_df = random_split(balanced_df, 0.7, 0.1)

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

################################################################################################
# Parte 2 - Carregando e configurando o modelo
################################################################################################

# Define o modelo a ser usado (GPT2 pequeno, 124M parâmetros)
CHOOSE_MODEL = "gpt2-small (124M)"

# Configuração base do modelo, incluindo tamanho do vocabulário, comprimento máximo do contexto,
# taxa de dropout, bias para qkv e arquitetura (embedding, camadas e cabeças)
BASE_CONFIG = {
    "vocab_size": 50257,        # tamanho do vocabulário GPT-2
    "context_length": 1024,     # tamanho máximo do contexto (número de tokens)
    "drop_rate": 0.0,           # dropout desabilitado para fine-tuning
    "qkv_bias": True,           # ativa bias nos cálculos query-key-value
    "emb_dim": 768,             # dimensão do embedding (GPT2-small)
    "n_layers": 12,             # número de camadas transformer (GPT2-small)
    "n_heads": 12,              # número de cabeças de atenção (GPT2-small)
}

# Extrai o tamanho do modelo (ex: "124M") do nome escolhido
model_size = CHOOSE_MODEL.split(" ")[-1].strip("()")

# Baixa os pesos do modelo pré-treinado e carrega os parâmetros
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2")

# Inicializa o modelo GPT2 com a configuração base
model = GPTModel(BASE_CONFIG)

# Carrega os pesos baixados para dentro do modelo
load_weights_into_gpt(model, params)

# Move o modelo para o dispositivo (GPU se disponível, senão CPU)
model.to(device)

# Define o modelo para modo de avaliação (desabilita dropout e batchnorm em modo treino)
model.eval()

# Definição do número de classes para classificação binária (ham/spam)
num_classes = 2

# Congela todos os parâmetros do modelo para não atualizá-los durante o treinamento
for param in model.parameters():
    param.requires_grad = False

# Substitui a camada final (cabeça do modelo) por uma nova camada linear para saída binária
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"],
                                 out_features=num_classes).to(device)

# Libera os parâmetros do último bloco transformer para treinamento (fine-tuning parcial)
for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

# Libera os parâmetros da normalização final (layer norm) para treinamento também
for param in model.final_norm.parameters():
    param.requires_grad = True

################################################################################################
# Parte 3 - Treinando o modelo
################################################################################################

# Cria o otimizador AdamW, que atualizará somente os parâmetros liberados para treino
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=5e-5, weight_decay=0.1)

# Número de épocas para o treinamento
num_epochs = 5

# Marca o tempo de início para medir a duração do treinamento
start_time = time.time()

# Executa o treinamento com a função customizada, retornando perdas, acurácias e contagem de exemplos vistos
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5)

# Marca o tempo de término do treinamento
end_time = time.time()

# Imprime o tempo total gasto no treinamento em minutos
print(f"Treinamento finalizado em {(end_time - start_time) / 60:.2f} minutos.")

################################################################################################
# Parte 4 - Validando o modelo
################################################################################################

# Plota os gráficos de perda (train e validation) ao longo do tempo/épocas
plot_values(torch.linspace(0, num_epochs, len(train_losses)),
            torch.linspace(0, examples_seen, len(train_losses)),
            train_losses, val_losses)

# Plota os gráficos de acurácia (train e validation) ao longo do tempo/épocas
plot_values(torch.linspace(0, num_epochs, len(train_accs)),
            torch.linspace(0, examples_seen, len(train_accs)),
            train_accs, val_accs, label="accuracy")

# Calcula a acurácia final para treino, validação e teste utilizando os loaders correspondentes
train_accuracy = calc_accuracy_loader(train_loader, model, device)
val_accuracy = calc_accuracy_loader(val_loader, model, device)
test_accuracy = calc_accuracy_loader(test_loader, model, device)

# Imprime as acurácias finais formatadas em porcentagem
print(f"Train Accuracy: {train_accuracy*100:.2f}%")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
