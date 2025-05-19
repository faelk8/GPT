import torch
import torch.nn as nn


# Define uma classe de atenção própria (Self-Attention)
class SelfAttention_v1(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        # Matrizes de projeção para query, key e value
        # São parâmetros treináveis com dimensões (d_in, d_out)
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # Calcula as queries, keys e values aplicando as projeções lineares
        # Aqui usamos multiplicação de matriz diretamente: (batch, d_in) @ (d_in, d_out)
        keys = self.W_key(x)      # shape: (seq_len, d_out)
        queries = self.W_query(x)    # shape: (seq_len, d_out)
        values = self.W_value(x)    # shape: (seq_len, d_out)

        # Calcula os "attention scores" fazendo o produto escalar entre queries e transposta de keys
        # shape: (seq_len, d_out) @ (d_out, seq_len) → (seq_len, seq_len)
        # matriz de similaridade entre todos os pares de tokens
        attn_scores = queries @ keys.T

        # Normaliza os scores com softmax (com escalamento pela raiz da dimensão dos keys)
        # Isso transforma os scores em pesos de atenção
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Multiplica os pesos de atenção pelos valores para obter o vetor de contexto
        # shape: (seq_len, seq_len) @ (seq_len, d_out) → (seq_len, d_out)
        context_vec = attn_weights @ values

        # Retorna os vetores de contexto (um para cada entrada)
        return context_vec
