import numpy as np
import torch
import torch.nn as nn
from transformer.transforme_block import TransformerBlock
from layer.layer_norm import LayerNorm


class GPTModel(nn.Module):
    """
    Implementa um modelo autoregressivo do tipo GPT (Generative Pretrained Transformer).

    Este modelo é baseado na arquitetura Transformer com blocos empilhados, usando
    embeddings de token e posição, camadas de atenção e projeção final para logits
    sobre o vocabulário.

    Parâmetros:
    -----------
    cfg : dict
        Dicionário de configuração contendo as seguintes chaves:
        - "vocab_size" (int): Tamanho do vocabulário (número total de tokens).
        - "emb_dim" (int): Dimensão dos embeddings e das camadas internas.
        - "context_length" (int): Comprimento máximo da sequência (janela de contexto).
        - "drop_rate" (float): Taxa de dropout aplicada em várias etapas.
        - "n_layers" (int): Número de blocos Transformer empilhados.
        - "n_heads" (int): Número de cabeças de atenção em cada bloco.
        - "qkv_bias" (bool): Se deve ou não incluir viés nas projeções QKV.

    Métodos:
    --------
    forward(in_idx: torch.LongTensor) -> torch.FloatTensor
        Executa o modelo GPT em uma sequência de índices de tokens.

    Parâmetros:
        in_idx : torch.LongTensor
            Tensor de entrada com forma (batch_size, seq_len) contendo os IDs dos tokens.

    Retorna:
        torch.FloatTensor
            Logits com forma (batch_size, seq_len, vocab_size), antes da aplicação de softmax.
    """

    def __init__(self, cfg):
        super().__init__()

        # Embedding de tokens (converte IDs em vetores)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])

        # Embedding de posições (para capturar a ordem dos tokens)
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        # Dropout aplicado após a soma dos embeddings
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Sequência de blocos Transformer (aqui apenas placeholders)
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Normalização final (também um placeholder)
        self.final_norm = LayerNorm(cfg["emb_dim"])

        # Cabeça de saída linear para prever logits sobre o vocabulário
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        """
        Executa o passo de inferência do modelo GPT.

        Esta função realiza o embedding dos tokens e suas posições, passa os dados
        pelos blocos Transformer empilhados, aplica a normalização final e projeta
        os vetores para o espaço do vocabulário para obter os logits.

        Parâmetros:
        -----------
        in_idx : torch.LongTensor
            Tensor com forma (batch_size, seq_len), contendo os índices dos tokens
            de entrada (IDs do vocabulário).

        Retorna:
        --------
        logits : torch.FloatTensor
            Tensor com forma (batch_size, seq_len, vocab_size), representando os
            logits não normalizados para cada posição da sequência e cada token
            do vocabulário. Pode ser usado diretamente com `F.cross_entropy` ou `softmax`.
        """
        batch_size, seq_len = in_idx.shape

        # Aplica o embedding de tokens e de posições
        tok_embeds = self.tok_emb(in_idx)  # (B, T, D)
        pos_embeds = self.pos_emb(torch.arange(
            seq_len, device=in_idx.device))  # (T, D)

        # Soma os embeddings e aplica dropout
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        # Passa pelos blocos Transformer (neste caso, não fazem nada)
        x = self.trf_blocks(x)

        # Normalização final (também não faz nada)
        x = self.final_norm(x)

        # Gera os logits finais
        logits = self.out_head(x)

        return logits
