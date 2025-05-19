import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        # Garante que a dimensão de saída possa ser dividida igualmente entre as cabeças
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # Dimensão interna de cada cabeça (porção do d_out)
        self.head_dim = d_out // num_heads

        # Projeções lineares para Q, K, V com dimensão de saída igual a d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # Camada linear final para projetar a concatenação das cabeças de volta para d_out
        self.out_proj = nn.Linear(d_out, d_out)

        self.dropout = nn.Dropout(dropout)

        # Máscara causal (superior triangular) para impedir que tokens "vejam o futuro"
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        # Projeta as entradas em Q, K e V de dimensão (b, num_tokens, d_out)
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Divide d_out em múltiplas cabeças: (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpõe para que a dimensão das cabeças venha antes dos tokens:
        # (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Produto escalar entre Q e K^T: (b, num_heads, num_tokens, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)

        # Aplica a máscara causal para impedir atenção ao futuro
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # Normaliza as pontuações com softmax e aplica dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Multiplica pesos de atenção por V: (b, num_heads, num_tokens, head_dim)
        context_vec = attn_weights @ values

        # Transpõe de volta para (b, num_tokens, num_heads, head_dim)
        context_vec = context_vec.transpose(1, 2)

        # Concatena as cabeças em uma única dimensão: (b, num_tokens, d_out)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)

        # Projeção final (opcional) para misturar as saídas das cabeças
        context_vec = self.out_proj(context_vec)

        return context_vec
