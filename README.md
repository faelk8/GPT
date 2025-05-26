



# 🔤 01 — Tokenização
Antes que um modelo de linguagem possa processar texto, é necessário transformar as palavras em tokens, ou seja, converter o texto em unidades menores que o modelo consegue entender e manipular. Isso é feito através de um tokenizer. Após o processamento, os tokens podem ser convertidos de volta para texto, permitindo reconstruir a saída gerada pelo modelo em linguagem natural.

### 🔖 Tokens especiais
Durante a tokenização, alguns tokens especiais são usados para indicar posições específicas em uma sequência de texto:

* [BOS] (Beginning of Sequence) — Marca o início da sequência. Esse token informa ao modelo onde o conteúdo começa.
* [EOS] (End of Sequence) — Indica o fim da sequência. Ele é útil especialmente quando se juntam vários textos, como artigos ou sentenças diferentes. Semelhante ao token <|endoftext|> utilizado no GPT-2.
* [PAD] (Padding) — Usado para preencher textos mais curtos durante o treinamento em lotes. Quando sequências possuem tamanhos diferentes, os textos mais curtos são estendidos com esse token para igualar o comprimento do lote.

### 🧬 Técnica de Tokenização — Byte Pair Encoding (BPE)
O método mais comum utilizado na tokenização de modelos como o GPT-2 é o Byte Pair Encoding (BPE). O BPE divide palavras em subpalavras ou caracteres com base em frequência de ocorrência. 
Palavras comuns são representadas por menos tokens, enquanto palavras raras são quebradas em partes menores.
Exemplo:
* A palavra desconectado pode ser tokenizada como: ["des", "con", "ect", "ado"].

Esse método é eficiente porque reduz o vocabulário necessário e permite ao modelo lidar com palavras desconhecidas ou inventadas de maneira mais robusta.

# 🧠 02 - Attention
Camada de Atenção (Attention Layer)
O modelo GPT-2 é baseado em uma arquitetura chamada Transformer, cujo principal componente é a camada de atenção. Essa camada permite que o modelo "preste atenção" em diferentes partes da entrada enquanto está processando uma palavra ou token, capturando relações de contexto de curto e longo alcance.

🧩 Como funciona a atenção?
Durante o treinamento, cada token da entrada é transformado em vetores chamados de query (Q), key (K) e value (V). A atenção é calculada comparando as queries de um token com as keys dos outros tokens — permitindo que o modelo decida quais palavras são mais relevantes para prever a próxima.

🎯 Multi-head Attention
Em vez de ter uma única "cabeça de atenção", o Transformer pode usar várias cabeças (como 2, 4, 8 ou mais). Cada cabeça aprende padrões diferentes de dependência entre palavras. Por exemplo:

* 1 cabeça: o modelo foca em um único padrão de contexto.
* 2 cabeças: o modelo pode aprender dois padrões distintos ao mesmo tempo, como relações sintáticas e semânticas.

No GPT-2, essa atenção multi-cabeça é uma das razões pela qual ele entende tão bem o contexto de uma frase, mesmo quando as palavras estão distantes.

⚙️ Exemplo prático
Se você estiver usando n_heads=2, o modelo divide o vetor de entrada em duas partes, aplica a atenção separadamente em cada uma, e depois concatena os resultados. Isso melhora a capacidade do modelo de capturar diferentes tipos de dependências linguísticas simultaneamente.

# 🧠 3 - O desafio de modelar sequências longas
Modelos de linguagem precisam lidar com sequências de texto de comprimento variável, mas muitos métodos tradicionais (como RNNs) sofrem com limitações no alcance de dependências longas — ou seja, perdem informações importantes quando os tokens estão muito distantes uns dos outros.<br>
O mecanismo de self-attention (captura dependências com mecanismos de atenção) resolve esse problema ao permitir que cada palavra "atenda" a todas as outras palavras da sequência, capturando relacionamentos globais entre os tokens de forma eficiente.

### 🔁 Self-Attention: Atenção a diferentes partes da entrada
No self-attention, cada token gera uma consulta (query), chave (key) e valor (value), permitindo o cálculo de pesos de atenção para todas as outras posições da sequência. Com isso, o modelo consegue:
* Focar nos tokens relevantes
* Ignorar informações irrelevantes
* Preservar contexto global em uma única camada

### ✅ Implementação simples sem pesos treináveis
Você pode implementar uma versão básica de self-attention que apenas calcula as similaridades entre os tokens, sem ajustar pesos.

### 📊 Cálculo de pesos de atenção
O cálculo dos pesos segue os seguintes passos:
* Multiplicação de Query e Key transposta
* Normalização por softmax
* Aplicação sobre os valores (Value)
* Geração da nova representação dos tokens

## ⚙️ Implementação com pesos treináveis
A versão mais completa da self-attention utiliza matrizes de pesos aprendíveis para transformar os tokens de entrada em Q, K e V. Isso torna o mecanismo mais expressivo e ajustável.

### 🕶️ Causal Attention: ocultando palavras futuras
Para tarefas de geração de texto, é importante que o modelo não veja palavras do futuro. Para isso usamos causal attention, onde aplicamos uma máscara triangular para bloquear os tokens à frente:
* Apenas tokens anteriores (ou o próprio) são considerados no cálculo da atenção.
* Essencial para preservar o comportamento autoregressivo do modelo.

### 🔒 Aplicação da máscara causal
O uso da máscara causal garante que:
* A saída no tempo t depende apenas de tokens até t.
* A previsão da próxima palavra não vaza informação futura.

### 🌧️ Dropout na atenção
Durante o treinamento, é comum aplicar dropout nos pesos de atenção, o que ajuda na regularização do modelo e evita overfitting.

### 🧩 Multi-Head Attention: atenção paralela
Em vez de usar uma única função de atenção, os modelos Transformers implementam multi-head attention:
* Dividem as representações em múltas "cabeças".
* Cada cabeça aprende padrões diferentes da sequência.
* Os resultados são concatenados e transformados em uma única saída.

### 🧱 Stacking de camadas de atenção
Empilhando várias camadas de multi-head attention, o modelo ganha profundidade e capacidade de abstração, aprendendo:
* Sintaxe (em camadas inferiores)
* Semântica (em camadas superiores)

### 📦 Implementação Compacta
As implementações modernas encapsulam a lógica de atenção em classes compactas, como SelfAttention, CausalSelfAttention e MultiHeadAttention, permitindo reutilização e legibilidade do código.


# 🔄 4 - Execução do Modelo GPT-2 com Tokenização e Geração de Texto
Este módulo demonstra como realizar a tokenização de frases, configurar um modelo GPT-2 com parâmetros específicos, e gerar novos textos a partir de um prompt inicial utilizando um modelo de linguagem (LLM) pré-treinado.

### 🧠 Tokenização
Textos de entrada são transformados em tokens numéricos utilizando o tokenizer compatível com o GPT-2. Esses tokens representam as palavras e subpalavras de forma que o modelo possa processá-los. São adicionados tokens especiais, como início e fim de sequência, e os textos são convertidos em tensores para posterior uso no modelo.

### ⚙️ Configuração do Modelo
É criada uma configuração baseada no GPT-2 de 124 milhões de parâmetros, contendo:
* Tamanho do vocabulário
* Comprimento máximo de contexto (número de tokens por entrada)
* Dimensão dos embeddings
* Número de camadas de Transformer e cabeças de atenção
* Taxa de dropout, entre outros

Essa configuração permite inicializar o modelo com as mesmas características do GPT-2 original.

### 🚀 Execução e Saída
O modelo é instanciado com os pesos definidos e executado sobre o batch de entrada, retornando uma matriz de logits — representações de probabilidade de cada próximo token possível para cada posição da sequência.

### ✍️ Geração de Texto
A partir de um prompt inicial, o modelo é capaz de prever a próxima palavra/token com base no contexto anterior. Um loop iterativo permite a geração de novos tokens até atingir um número máximo ou um token de parada. O resultado final é decodificado de volta para texto compreensível.

Essa etapa mostra o pipeline completo de entrada, processamento e geração de saída textual, simulando o comportamento básico de um modelo de linguagem autoregressivo como o GPT-2.

# ✉️ 06 - Spam
Detecção de Spam com LLM (GPT-2 Fine-Tuned)
Este projeto utiliza um modelo de linguagem grande (LLM) baseado no GPT-2 para a tarefa de detecção de spam. Combinando a capacidade contextual dos modelos de linguagem com uma camada de classificação supervisionada, é possível transformar o GPT-2 em um poderoso classificador binário (spam ou não spam).

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

# 💪 07 - Intruções para Treinamento
Este projeto realiza o fine-tuning de um modelo GPT-2 (ex: gpt2-small, gpt2-medium, etc.) para a tarefa de pergunta e resposta baseada em instruções (Instruction Tuning).

Ao contrário do treinamento tradicional que se baseia apenas na predição da próxima palavra em sequências de texto, aqui utilizei um formato estruturado com instruções explícitas. Esse formato guia o modelo a entender e seguir comandos, tornando-o mais eficaz para tarefas práticas como assistentes virtuais, automação de suporte, geração de conteúdo e muito mais.

* O processo de ajuste fino de instruções adapta um LLM pré-treinado para seguir instruções humanas e gerar as respostas desejadas.
* A preparação do conjunto de dados envolve o download de um conjunto de dados de instruções-respostas, a formatação das entradas e a divisão em conjuntos de treinamento, validação e teste.
* Os lotes de treinamento são construídos usando uma função de agrupamento personalizada que preenche sequências, cria IDs de tokens de destino e mascara tokens de preenchimento.
* Carregamos um modelo GPT-2 pré-treinado com 124 milhões de parâmetros para servir como ponto de partida para o ajuste fino das instruções.
* O modelo pré-treinado é ajustado no conjunto de dados de instruções usando um loop de treinamento semelhante ao pré-treinamento.
* A avaliação envolve a extração das respostas do modelo em um conjunto de teste e sua pontuação (por exemplo, usando outro LLM).




# Referência
Build a Large Language Model - Sebastian Raschka<br>
Livros gratuitos para download: [Projeto Gutenberg](https://www.gutenberg.org/browse/languages/pt)<br>
Conjunto de dados utilizado para treinamento com pergunta e respostas: [Alpaca dataset - Stanford](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json)

