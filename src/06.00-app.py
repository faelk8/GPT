import tiktoken
import torch
import chainlit

from models.gpt_model import GPTModel
from spam.spam_functions import classify_review

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Carrega o modelo GPT-2 fine-tunado e seu tokenizer.

    Esta função instancia a arquitetura do modelo GPT-2 com configuração equivalente ao GPT-2 124M,
    ajusta a saída para ser usada como classificador (com 2 classes) e carrega os pesos salvos 
    do modelo fine-tunado localizado no caminho 'save/review_classifier.pth'.

    Requisitos:
    -----------
    - O arquivo de pesos 'review_classifier.pth' precisa ter sido gerado previamente conforme o capítulo 6.
    - A biblioteca 'tiktoken' precisa estar instalada para o uso do tokenizer.

    Retorna:
    --------
    tokenizer : tiktoken.Encoding
        Tokenizador compatível com o modelo GPT-2.
    model : GPTModel
        Modelo GPT-2 carregado com pesos fine-tunados, ajustado como classificador binário.
    """

    # Define a configuração do modelo GPT-2 (tamanho 124M)
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Tamanho do vocabulário GPT-2
        "context_length": 1024,  # Comprimento máximo do contexto (tokens)
        "emb_dim": 768,          # Dimensão do embedding
        "n_heads": 12,           # Número de cabeças de atenção
        "n_layers": 12,          # Número de camadas do Transformer
        "drop_rate": 0.1,        # Taxa de dropout
        "qkv_bias": True         # Usar bias nas projeções QKV
    }

    # Obtém o tokenizer padrão do GPT-2 (usando tiktoken)
    tokenizer = tiktoken.get_encoding("gpt2")

    # Caminho do arquivo contendo os pesos do modelo fine-tunado
    model_path = "save/review_classifier.pth"

    # Instancia o modelo com a configuração especificada
    model = GPTModel(GPT_CONFIG_124M)

    # Altera a cabeça de saída do modelo para um classificador binário (duas classes)
    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)

    # Carrega os pesos do modelo a partir do checkpoint salvo
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)

    # Move o modelo para o dispositivo adequado (GPU ou CPU)
    model.to(device)

    # Coloca o modelo em modo de avaliação (desativa dropout, batchnorm, etc.)
    model.eval()

    # Retorna o tokenizer e o modelo carregado
    return tokenizer, model


# Carrega o tokenizer e o modelo para uso posterior no chat (função Chainlit)
tokenizer, model = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    Função principal que será chamada sempre que uma nova mensagem for recebida no Chainlit.

    Parâmetros:
    -----------
    message : chainlit.Message
        Mensagem recebida do usuário via interface do Chainlit.
    """

    # Obtém o texto da mensagem enviada pelo usuário
    user_input = message.content

    # Classifica a review como "spam" ou "not spam"
    label = classify_review(
        user_input, model, tokenizer, device, max_length=120)

    # Envia a resposta do modelo de volta à interface do usuário
    await chainlit.Message(
        content=f"{label}",
    ).send()
