import re
import json
import tiktoken
import torch
import time
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm

from download.gpt_download import download_and_load_gpt2
from models.gpt_model import GPTModel
from weights.weights import load_weights_into_gpt

from function.geral import generate, text_to_token_ids, token_ids_to_text, custom_collate_fn
from metrics.metrics import calc_loss_loader_instruction
from instructions.instruction_dataset import InstructionDataset
from train.train_model import train_model_simple
from plots.plots import plot_losses

torch.manual_seed(123)

##################################################################################
# Parte 1 - Carregando os dados e preparando os dados
##################################################################################
file_path = "data/instruction-data.json"
with open(file_path, "r", encoding="utf-8") as file:
    data = json.load(file)


def format_input(entry):
    """
    Formata uma entrada de dado contendo uma instrução e, opcionalmente, um input adicional.

    Parâmetros:
    -----------
    entry : dict
        Um dicionário contendo as chaves:
        - 'instruction': a tarefa que o modelo deve realizar.
        - 'input': um contexto ou dado extra para ajudar na geração (pode ser uma string vazia).

    Retorna:
    --------
    str
        Uma string formatada que combina a instrução e o input (caso exista),
        pronta para ser usada como prompt de entrada para o modelo.
    """

    # Texto base com a instrução
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # Adiciona o campo 'input' caso ele não seja vazio
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    # Concatena instrução e input (se houver)
    return instruction_text + input_text


# Divide o conjunto de dados original em 3 partes:
train_portion = int(len(data) * 0.85)  # 85% dos dados para treinamento
test_portion = int(len(data) * 0.1)    # 10% dos dados para teste
# Os 5% restantes vão para validação
val_portion = len(data) - train_portion - test_portion

# Fatia os dados conforme os índices definidos acima
train_data = data[:train_portion]  # Pega os primeiros 85%
test_data = data[train_portion:train_portion +
                 test_portion]  # Pega os 10% seguintes
val_data = data[train_portion + test_portion:]  # O restante (últimos 5%)

# Carrega o tokenizador 'gpt2' usando tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

# Define o dispositivo como GPU (cuda) se disponível, senão CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define novamente o dispositivo (redundante, pode remover uma das linhas acima)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cria uma função de collation customizada, fixando os parâmetros device e tamanho máximo
customized_collate_fn = partial(
    custom_collate_fn,            # Função que prepara os batches
    device=device,                # Move os tensores para o dispositivo correto
    allowed_max_length=1024      # Limita o comprimento máximo dos tokens por amostra
)

# Define número de subprocessos para carregar dados (0 = main thread, útil para debug ou compatibilidade)
num_workers = 0
# Define o tamanho do batch para treino/validação/teste
batch_size = 8

# Cria dataset de treinamento usando a classe InstructionDataset
train_dataset = InstructionDataset(train_data, tokenizer)
# Cria o DataLoader de treino, que gera batches de dados aleatórios (shuffle=True)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,  # Usa função de collation definida
    shuffle=True,                      # Embaralha os dados a cada epoch
    drop_last=True,                    # Descarta último batch se estiver incompleto
    num_workers=num_workers
)

# Cria dataset de validação
val_dataset = InstructionDataset(val_data, tokenizer)
# Cria o DataLoader de validação (sem shuffle)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,                     # Mantém a ordem original dos dados
    drop_last=False,                   # Não descarta o último batch, mesmo que incompleto
    num_workers=num_workers
)

# Cria dataset de teste
test_dataset = InstructionDataset(test_data, tokenizer)
# Cria o DataLoader de teste (sem shuffle)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)


##################################################################################
# Parte 2 - Carregando o modelo
##################################################################################

# Define configurações base do modelo
BASE_CONFIG = {
    "vocab_size": 50257,     # Tamanho do vocabulário (GPT-2 usa 50.257 tokens)
    # Número máximo de tokens que o modelo pode considerar de contexto
    "context_length": 1024,
    # Taxa de dropout (0.0 significa sem regularização)
    "drop_rate": 0.0,
    "qkv_bias": True         # Usa viés nos vetores Q, K e V do Transformer
}

# Dicionário com configurações específicas para diferentes tamanhos do GPT-2
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Define qual modelo será utilizado
CHOOSE_MODEL = "gpt2-small (124M)"

# Atualiza BASE_CONFIG com os parâmetros do modelo escolhido
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# Extrai o tamanho do modelo do nome para uso na função de download
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

# Baixa os arquivos de pesos e carrega os parâmetros do modelo
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"  # Diretório onde os modelos estão armazenados
)

# Inicializa a arquitetura do modelo com base nas configurações
model = GPTModel(BASE_CONFIG)

# Carrega os pesos treinados no modelo inicializado
load_weights_into_gpt(model, params)

# Coloca o modelo em modo de avaliação (sem dropout, sem gradientes)
model.eval()

# Prepara o texto de entrada com base em uma amostra de validação
input_text = format_input(val_data[0])

# Converte texto para tokens e gera uma continuação
token_ids = generate(
    model=model,
    # Tokeniza o texto de entrada
    idx=text_to_token_ids(input_text, tokenizer),
    # Número máximo de tokens a serem gerados
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],    # Contexto máximo suportado
    eos_id=50256,                                  # ID do token de parada
)

# Converte os tokens gerados de volta para texto
generated_text = token_ids_to_text(token_ids, tokenizer)

# Extrai apenas a resposta gerada (remove o texto original e rótulo "### Response:")
response_text = (
    generated_text[len(input_text):]
    .replace("### Response:", "")
    .strip()
)

# Move o modelo para o dispositivo apropriado (CPU ou GPU)
model.to(device)

# Avalia o modelo em batches de treino e validação (modo sem gradientes)
with torch.no_grad():
    train_loss = calc_loss_loader_instruction(
        train_loader, model, device, num_batches=5)
    val_loss = calc_loss_loader_instruction(
        val_loader, model, device, num_batches=5)

# Imprime as perdas médias em treino e validação
print()
print("Training loss:", train_loss)
print("Validation loss:", val_loss)
print()


##################################################################################
# Parte 3 - Finetuning
##################################################################################

# Marca o tempo de início do treinamento
start_time = time.time()

# Define o otimizador AdamW com taxa de aprendizado baixa e regularização (weight decay)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

# Número de épocas de treinamento
num_epochs = 10

# Inicia o treinamento do modelo
# - model: o modelo GPT inicializado com os pesos base
# - train_loader / val_loader: dados de treino e validação
# - optimizer: otimizador definido
# - device: CPU ou GPU
# - num_epochs: número de épocas para treinar
# - eval_freq: frequência com que avalia durante treino (a cada 5 épocas)
# - eval_iter: número de batches usados na avaliação
# - start_context: contexto de entrada padrão para geração
# - tokenizer: para tokenização dos textos
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=format_input(val_data[0]), tokenizer=tokenizer
)

# Marca o tempo de término e calcula a duração em minutos
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# Cria um tensor com os valores das épocas para plotar as perdas
epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

# Plota as curvas de perda de treino e validação ao longo das épocas
plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


# Para os 3 primeiros exemplos do conjunto de teste
for entry in test_data[:3]:

    # Formata a entrada no padrão esperado pelo modelo (Instruction + Input)
    input_text = format_input(entry)

    # Gera resposta baseada no modelo já treinado
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(
            device),  # Tokeniza e move para GPU/CPU
        max_new_tokens=256,                                       # Gera até 256 tokens
        # Tamanho máximo do contexto
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256                                              # Token de parada
    )

    # Converte os tokens de volta para texto legível
    generated_text = token_ids_to_text(token_ids, tokenizer)

    # Remove a entrada original e a tag "### Response:" para extrair apenas a resposta
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
    )

    # Imprime a entrada formatada, a resposta esperada e a resposta do modelo
    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")


# Para todos os exemplos no test_data, usando tqdm para barra de progresso
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

    # Formata a entrada
    input_text = format_input(entry)

    # Gera os tokens de resposta
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )

    # Converte tokens em texto e extrai apenas a resposta gerada
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace(
        "### Response:", "").strip()

    # Armazena a resposta gerada dentro do próprio dicionário do exemplo
    test_data[i]["model_response"] = response_text


# Salva os dados de teste com as respostas do modelo em JSON
# with open("instruction-data-with-response.json", "w") as file:
#     json.dump(test_data, file, indent=4)  # Usa indentação para facilitar leitura

# Salva os pesos do modelo treinado como um arquivo `.pth`
# file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
# torch.save(model.state_dict(), file_name)
# print(f"Model saved as {file_name}")
