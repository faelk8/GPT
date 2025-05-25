import torch


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    Classifica uma avaliação textual como "spam" ou "not spam" usando um modelo Transformer.

    Esta função prepara o texto de entrada, codificando-o com um tokenizer e ajustando o tamanho
    da sequência com truncamento e padding. Em seguida, ela realiza a inferência com o modelo
    fornecido e retorna a classe predita com base no último token gerado.

    Parâmetros:
    -----------
    text : str
        Texto da avaliação a ser classificada.

    model : nn.Module
        Modelo de linguagem treinado para classificação (ex: variante de GPT).

    tokenizer : Callable
        Tokenizador que converte o texto em uma sequência de IDs de tokens.

    device : torch.device
        Dispositivo onde o modelo está carregado (ex: 'cuda' ou 'cpu').

    max_length : int, opcional
        Tamanho máximo da sequência de entrada. A sequência será truncada e/ou
        preenchida com padding para atingir esse comprimento.

    pad_token_id : int, opcional (default = 50256)
        ID do token de padding utilizado para preencher a sequência até `max_length`.

    Retorno:
    --------
    str
        Retorna "spam" se a classe prevista for 1, e "not spam" se for 0.
    """
    model.eval()

    # Codifica o texto em IDs de tokens
    input_ids = tokenizer.encode(text)

    # Determina o comprimento máximo suportado pelo modelo (normalmente 1024 ou 512)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Trunca a sequência para o comprimento permitido
    input_ids = input_ids[:min(max_length, supported_context_length)]

    # Adiciona padding à direita até o comprimento máximo
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    # Converte para tensor e adiciona dimensão de batch
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # Desativa o cálculo de gradiente para inferência
    with torch.no_grad():
        # Obtém os logits do último token da sequência
        logits = model(input_tensor)[:, -1, :]

    # Prediz a classe com maior valor de logit
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"
