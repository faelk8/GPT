



# 01.01-Token
Transformando palavras em Tokens e convertendo de volta incluido Token que marca o início e fim da frase.

* [BOS] (início da sequência) — Este token marca o início de um texto. Ele indica para o LLM onde um trecho de conteúdo começa.<br>
* [EOS] (fim da sequência) — Este token é posicionado no final de um texto e é especialmente útil ao concatenar vários textos não relacionados, semelhante a <|endoftext|>. Por exemplo, ao  dois artigos ou livros diferentes da Wikipédia, o token [EOS] indica onde um termina e o próximo começa.<br>
* [PAD] (preenchimento) — Ao treinar LLMs com tamanhos de lote maiores que um, o lote pode conter textos de tamanhos variados. Para garantir que todos os textos tenham o mesmo comprimento, os textos mais curtos são estendidos ou "preenchidos" usando o token [PAD], até o comprimento do texto mais longo do lote.<br>

A técina mais utilizada no momento é byte pair econding.

BPE quebra palavras individuais em caracteres.


# 02.01-Attention
Camada de attention que pode ser uma cabeça ou 2 cabeças.

# 06 - Spam
Detecção de spam com modelo pré treinados utilizando LLM.

* Preparando o conjunto de dados
* Criando um carregador de dados
* Inicializando um modelo com pesos pré-treinado
* Adicionando uma classification head
* Cálculo da perda e precisão da classificação
* Ajustando o modelo em dados supervisionados
* Usando o LLM como um classificador de spam

`Para testar com interface visual.`
```python
chainlit run 06.00-app.py
```

# Referência

Livros gratuitos para download: [Projeto Gutenberg](https://www.gutenberg.org/browse/languages/pt)