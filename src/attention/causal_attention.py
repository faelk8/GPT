import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()

        self.d_out = d_out

        # Projeções lineares para obter queries, keys e values a partir do input x
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Dropout aplicado nos pesos de atenção para regularização
        self.dropout = nn.Dropout(dropout)

        # Máscara triangular superior usada para bloquear posições futuras (atenção causal)
        # Register_buffer mantém o tensor como parte do estado do módulo sem ser um parâmetro treinável
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        # x tem shape [batch_size, num_tokens, d_in]
        b, num_tokens, d_in = x.shape

        # Projeções lineares
        keys = self.W_key(x)      # shape: [b, num_tokens, d_out]
        queries = self.W_query(x)  # shape: [b, num_tokens, d_out]
        values = self.W_value(x)  # shape: [b, num_tokens, d_out]

        # Calcula pontuações de atenção com produto escalar entre queries e transposição de keys
        # shape: [b, num_tokens, num_tokens]
        attn_scores = queries @ keys.transpose(1, 2)

        # Aplica a máscara causal para evitar que posições acessem tokens futuros
        # A máscara define -inf para as posições superiores à diagonal
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )

        # Normaliza as pontuações com softmax (dividido pela raiz de d_k para estabilidade)
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        # Aplica dropout nos pesos de atenção
        attn_weights = self.dropout(attn_weights)

        # Combina os valores com os pesos de atenção para obter os vetores de contexto
        context_vec = attn_weights @ values  # shape: [b, num_tokens, d_out]
        return context_vec
