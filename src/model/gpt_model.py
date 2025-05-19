import torch
import torch.nn as nn


class GPTModel(nn.Module):
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
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Normalização final (também um placeholder)
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])

        # Cabeça de saída linear para prever logits sobre o vocabulário
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
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


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Placeholder para um bloco Transformer real
        pass

    def forward(self, x):
        # Apenas retorna a entrada (não faz nada)
        return x


class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        # Placeholder para LayerNorm real
        pass

    def forward(self, x):
        # Apenas retorna a entrada (não faz nada)
        return x
