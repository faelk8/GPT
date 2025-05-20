# Importa a classe GPTModel do módulo onde o modelo está implementado
import numpy as np
from src.core.gpt_model import GPTModel
from src.fuction.geral import generate_text_simple, create_dataloader_v1, load_text_file_from_url_if_needed, generate_temperature
# Importa a biblioteca PyTorch
import torch
import tiktoken
import os


# Carrega arquivo de texto localmente, ou faz download se não existir
file_path = "data/the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
text_data = load_text_file_from_url_if_needed(file_path, url)


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={
                               '<|endoftext|>'})  # Codifica o texto
    # Converte em tensor e adiciona dimensão de batch
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

# Função para converter IDs de tokens de volta para texto


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # Remove dimensão de batch
    return tokenizer.decode(flat.tolist())  # Decodifica os tokens para texto


start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")


def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(
        open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)


# Dicionário base de configuração do modelo GPT-2 pequeno (124M)
GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Tamanho do vocabulário
    # Comprimento do contexto (número de tokens que o modelo pode "ver" por vez); original era 1024
    "context_length": 256,
    # Dimensão do embedding (tamanho do vetor que representa cada token)
    "emb_dim": 768,
    "n_heads": 12,           # Número de cabeças de atenção
    "n_layers": 12,          # Número de camadas do Transformer
    "drop_rate": 0.1,        # Taxa de dropout para regularização
    "qkv_bias": False        # Se os vetores Q, K e V devem ter viés (bias)
}

# Dicionário com configurações específicas para diferentes variantes do GPT-2
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Escolha do modelo a ser carregado — você pode mudar aqui para outro modelo listado acima
model_name = "gpt2-small (124M)"  # Nome do modelo escolhido

# Copia a configuração base
NEW_CONFIG = GPT_CONFIG_124M.copy()

# Atualiza a configuração com os parâmetros específicos do modelo escolhido
NEW_CONFIG.update(model_configs[model_name])

# Atualiza outras configurações específicas, como contexto maior e uso de bias em QKV
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

# Instancia o modelo com a configuração final
gpt = GPTModel(NEW_CONFIG)

# Define o dispositivo para execução: GPU se disponível, caso contrário CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move o modelo para o dispositivo selecionado (GPU ou CPU)
gpt.to(device)

# Carrega os pesos do modelo a partir de um arquivo salvo no caminho especificado
gpt.load_state_dict(torch.load(
    "src/models/gpt2-small-124M.pth", map_location=device))

# (ERRO CORRIGIDO) `model` não está definido. Substituí por `gpt.eval()` diretamente.
gpt.eval()  # Coloca o modelo em modo de avaliação (desativa dropout, etc.)


def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


load_weights_into_gpt(gpt, params)
gpt.to(device)
