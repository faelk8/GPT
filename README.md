
<a href="#"></a>

<div style="text-align: center;">
  <h1>GPT</h1>
</div>


# 🧠 Sobre este Repositório
Este repositório é um guia estruturado e prático sobre os fundamentos e a implementação do modelo GPT (Generative Pre-trained Transformer). Nele, exploramos desde os conceitos iniciais como tokenização e tokens especiais, até os componentes centrais da arquitetura Transformer, como self-attention, multi-head attention e máscara causal.

Além disso, demonstramos a execução do GPT-2 com exemplos de tokenização, geração de texto, e estratégias de avaliação e treinamento de modelos generativos, incluindo o uso de pesos pré-treinados da OpenAI.

## 📚 Menu
<ul>
  <li><a href="#01">🔤 01 - Tokenização</a>
    <ul>
      <li><a href="#01.01">🔖 Tokens especiais</a></li>
      <li><a href="#01.02">🔖 🧬 Técnica de Tokenização — Byte Pair Encoding (BPE)</a></li>
    </ul>
  </li>

  <li><a href="#02">🧠 02 - Attention</a>
    <ul>
      <li><a href="#02.01">🧩 Como funciona a atenção?</a></li>
      <li><a href="#02.02">🎯 Multi-head Attention</a></li>
      <li><a href="#02.03">⚙️ Exemplo prático</a></li>
    </ul>
  </li>

  <li><a href="#03">🧠 3 - Attention o desafio de modelar sequências longas</a>
    <ul>
      <li><a href="#03.01">🔁 Self-Attention: Atenção a diferentes partes da entrada</a></li>
      <li><a href="#03.02">✅ Implementação simples sem pesos treináveis</a></li>
      <li><a href="#03.03">📊 Cálculo de pesos de atenção</a></li>
      <li><a href="#03.04">⚙️ Implementação com pesos treináveis</a></li>
      <li><a href="#03.05">🕶️ Causal Attention: ocultando palavras futuras</a></li>
      <li><a href="#03.06">🔒 Aplicação da máscara causal</a></li>
      <li><a href="#03.07">🌧️ Dropout na atenção</a></li>
      <li><a href="#03.08">🧩 Multi-Head Attention: atenção paralela</a></li>
      <li><a href="#03.09">🧱 Stacking de camadas de atenção</a></li>
      <li><a href="#03.10">📦 Implementação Compacta</a></li>
    </ul>
  </li>

  <li><a href="#04">🔄 4 - Execução do Modelo GPT-2 com Tokenização e Geração de Texto</a>
    <ul>
      <li><a href="#04.01">🧠 Tokenização</a></li>
      <li><a href="#04.02">⚙️ Configuração do Modelo</a></li>
      <li><a href="#04.03">🚀 Execução e Saída</a></li>
      <li><a href="#04.04">✍️ Geração de Texto</a></li>
    </ul>
  </li>

  <li><a href="#05">✨ Avaliação e Treinamento de Modelos de Texto Generativo</a>
    <ul>
      <li><a href="#05.01">⭐ Avaliação de Modelos de Texto Generativo</a></li>
      <li><a href="#05.02">💪 Treinamento de um LLM (Large Language Model)</a></li>
      <li><a href="#05.03">♟️ Estratégias de Decodificação para Controlar Aleatoriedade</a></li>
      <li><a href="#05.04">🏗️ Carregamento de Pesos Pré-Treinados da OpenAI</a></li>
    </ul>
  </li>

  <li><a href="#06">✉️ 06 - Spam</a></li>
  <li><a href="#07">💪 07 - Instruções para Treinamento</a></li>
  <li><a href="#08">Referência</a></li>
</ul>


##

<h1 id="01">🔤 1 - Tokenização</h1>
Antes que um modelo de linguagem possa processar texto, é necessário transformar as palavras em tokens, ou seja, converter o texto em unidades menores que o modelo consegue entender e manipular. Isso é feito através de um tokenizer. Após o processamento, os tokens podem ser convertidos de volta para texto, permitindo reconstruir a saída gerada pelo modelo em linguagem natural.

<h3 id="01.01">🔖 Tokens especiais </h3> 
Durante a tokenização, alguns tokens especiais são usados para indicar posições específicas em uma sequência de texto:

* [BOS] (Beginning of Sequence) — Marca o início da sequência. Esse token informa ao modelo onde o conteúdo começa.
* [EOS] (End of Sequence) — Indica o fim da sequência. Ele é útil especialmente quando se juntam vários textos, como artigos ou sentenças diferentes. Semelhante ao token <|endoftext|> utilizado no GPT-2.
* [PAD] (Padding) — Usado para preencher textos mais curtos durante o treinamento em lotes. Quando sequências possuem tamanhos diferentes, os textos mais curtos são estendidos com esse token para igualar o comprimento do lote.

<h3 id="01.02">🧬 Técnica de Tokenização — Byte Pair Encoding (BPE)</h3> 
O método mais comum utilizado na tokenização de modelos como o GPT-2 é o Byte Pair Encoding (BPE). O BPE divide palavras em subpalavras ou caracteres com base em frequência de ocorrência. 
Palavras comuns são representadas por menos tokens, enquanto palavras raras são quebradas em partes menores.
Exemplo:
* A palavra desconectado pode ser tokenizada como: ["des", "con", "ect", "ado"].

Esse método é eficiente porque reduz o vocabulário necessário e permite ao modelo lidar com palavras desconhecidas ou inventadas de maneira mais robusta.

<div style="text-align: right;">
  <a href="#">🔝 Voltar ao topo</a>
</div>

<h1 id="02">🧠 2 - Attention</h1>
Camada de Atenção (Attention Layer)
O modelo GPT-2 é baseado em uma arquitetura chamada Transformer, cujo principal componente é a camada de atenção. Essa camada permite que o modelo "preste atenção" em diferentes partes da entrada enquanto está processando uma palavra ou token, capturando relações de contexto de curto e longo alcance.

<h3 id="02.01">🧩 Como funciona a atenção?</h3> 
Durante o treinamento, cada token da entrada é transformado em vetores chamados de query (Q), key (K) e value (V). A atenção é calculada comparando as queries de um token com as keys dos outros tokens — permitindo que o modelo decida quais palavras são mais relevantes para prever a próxima.

<h3 id="02.02">🎯 Multi-head Attention</h3> 
Em vez de ter uma única "cabeça de atenção", o Transformer pode usar várias cabeças (como 2, 4, 8 ou mais). Cada cabeça aprende padrões diferentes de dependência entre palavras. Por exemplo:

* 1 cabeça: o modelo foca em um único padrão de contexto.
* 2 cabeças: o modelo pode aprender dois padrões distintos ao mesmo tempo, como relações sintáticas e semânticas.

No GPT-2, essa atenção multi-cabeça é uma das razões pela qual ele entende tão bem o contexto de uma frase, mesmo quando as palavras estão distantes.

<h3 id="02.03">⚙️ Exemplo prático</h3> </h3> 
Se você estiver usando n_heads=2, o modelo divide o vetor de entrada em duas partes, aplica a atenção separadamente em cada uma, e depois concatena os resultados. Isso melhora a capacidade do modelo de capturar diferentes tipos de dependências linguísticas simultaneamente.

<div style="text-align: right;">
  <a href="#">🔝 Voltar ao topo</a>
</div>


<h1 id="03">🧠 3 - Attention o desafio de modelar sequências longas</h1>
Modelos de linguagem precisam lidar com sequências de texto de comprimento variável, mas muitos métodos tradicionais (como RNNs) sofrem com limitações no alcance de dependências longas — ou seja, perdem informações importantes quando os tokens estão muito distantes uns dos outros.<br>
O mecanismo de self-attention (captura dependências com mecanismos de atenção) resolve esse problema ao permitir que cada palavra "atenda" a todas as outras palavras da sequência, capturando relacionamentos globais entre os tokens de forma eficiente.
No self-attention, cada token gera uma consulta (query), chave (key) e valor (value), permitindo o cálculo de pesos de atenção para todas as outras posições da sequência. Com isso, o modelo consegue:
* Focar nos tokens relevantes
* Ignorar informações irrelevantes
* Preservar contexto global em uma única camada

<h3 id="03.02">✅ Implementação simples sem pesos treináveis</h3> 
Você pode implementar uma versão básica de self-attention que apenas calcula as similaridades entre os tokens, sem ajustar pesos.

<h3 id="03.03">📊 Cálculo de pesos de atenção</h3> 
O cálculo dos pesos segue os seguintes passos:
* Multiplicação de Query e Key transposta
* Normalização por softmax
* Aplicação sobre os valores (Value)
* Geração da nova representação dos tokens

<h3 id="03.04">⚙️ Implementação com pesos treináveis</h3>
A versão mais completa da self-attention utiliza matrizes de pesos aprendíveis para transformar os tokens de entrada em Q, K e V. Isso torna o mecanismo mais expressivo e ajustável.

<h3 id="03.04">🕶️ Causal Attention: ocultando palavras futuras</h3> 
Para tarefas de geração de texto, é importante que o modelo não veja palavras do futuro. Para isso usamos causal attention, onde aplicamos uma máscara triangular para bloquear os tokens à frente:
* Apenas tokens anteriores (ou o próprio) são considerados no cálculo da atenção.
* Essencial para preservar o comportamento autoregressivo do modelo.

<h3 id="03.05">🔒 Aplicação da máscara causal</h3> 
O uso da máscara causal garante que:
* A saída no tempo t depende apenas de tokens até t.
* A previsão da próxima palavra não vaza informação futura.

<h3 id="03.05">🌧️ Dropout na atenção</h3> 
Durante o treinamento, é comum aplicar dropout nos pesos de atenção, o que ajuda na regularização do modelo e evita overfitting.

<h3 id="03.06">🧩 Multi-Head Attention: atenção paralela</h3> 
Em vez de usar uma única função de atenção, os modelos Transformers implementam multi-head attention:
* Dividem as representações em múltas "cabeças".
* Cada cabeça aprende padrões diferentes da sequência.
* Os resultados são concatenados e transformados em uma única saída.

<h3 id="03.07">🧱 Stacking de camadas de atenção</h3>
Empilhando várias camadas de multi-head attention, o modelo ganha profundidade e capacidade de abstração, aprendendo:
* Sintaxe (em camadas inferiores)
* Semântica (em camadas superiores)

<h3 id="03.08">📦 Implementação Compacta</h3> 
As implementações modernas encapsulam a lógica de atenção em classes compactas, como SelfAttention, CausalSelfAttention e MultiHeadAttention, permitindo reutilização e legibilidade do código.

<div style="text-align: right;">
  <a href="#">🔝 Voltar ao topo</a>
</div>

<h1 id="04">🔄 4 - Execução do Modelo GPT-2 com Tokenização e Geração de Texto</h1>
Este módulo demonstra como realizar a tokenização de frases, configurar um modelo GPT-2 com parâmetros específicos, e gerar novos textos a partir de um prompt inicial utilizando um modelo de linguagem (LLM) pré-treinado.

<h3 id="04.01">🧠 Tokenização</h3> 
Textos de entrada são transformados em tokens numéricos utilizando o tokenizer compatível com o GPT-2. Esses tokens representam as palavras e subpalavras de forma que o modelo possa processá-los. São adicionados tokens especiais, como início e fim de sequência, e os textos são convertidos em tensores para posterior uso no modelo.

<h3 id="04.02">⚙️ Configuração do Modelo</h3> 
É criada uma configuração baseada no GPT-2 de 124 milhões de parâmetros, contendo:
* Tamanho do vocabulário
* Comprimento máximo de contexto (número de tokens por entrada)
* Dimensão dos embeddings
* Número de camadas de Transformer e cabeças de atenção
* Taxa de dropout, entre outros

Essa configuração permite inicializar o modelo com as mesmas características do GPT-2 original.

<h3 id="04.03">🚀 Execução e Saída</h3> 
O modelo é instanciado com os pesos definidos e executado sobre o batch de entrada, retornando uma matriz de logits — representações de probabilidade de cada próximo token possível para cada posição da sequência.

<h3 id="04.05">✍️ Geração de Texto</h3> 
A partir de um prompt inicial, o modelo é capaz de prever a próxima palavra/token com base no contexto anterior. Um loop iterativo permite a geração de novos tokens até atingir um número máximo ou um token de parada. O resultado final é decodificado de volta para texto compreensível.

Essa etapa mostra o pipeline completo de entrada, processamento e geração de saída textual, simulando o comportamento básico de um modelo de linguagem autoregressivo como o GPT-2.

<div style="text-align: right;">
  <a href="#">🔝 Voltar ao topo</a>
</div>


<h1 id="05">✨ 5 - Avaliação e Treinamento de Modelos de Texto Generativo</h1>
Esta etapa do projeto explora como construir, avaliar, treinar e personalizar modelos de linguagem baseados na arquitetura GPT (Generative Pretrained Transformer). A proposta é trabalhar desde a geração de texto com modelos simples até o uso de modelos pré-treinados, passando por estratégias de decodificação e técnicas de ajuste fino.

<h3 id="05.01">⭐ Avaliação de Modelos de Texto Generativo</h3> 
Nesta seção, é demonstrado como gerar texto automaticamente a partir de um prompt inicial utilizando um modelo de linguagem. O processo consiste em fornecer uma frase inicial e deixar o modelo prever os próximos tokens. A avaliação do desempenho pode ser feita calculando a perda (loss) associada às previsões, além de comparar os resultados gerados com os dados reais.

<h3 id="05.02">💪 Treinamento de um LLM (Large Language Model)</h3> 
Esta parte mostra como treinar um modelo GPT a partir do zero. Para isso, é feita a preparação dos dados com divisão entre conjuntos de treinamento e validação, configuração dos hiperparâmetros do modelo e uso de otimizadores adequados. São utilizadas versões ajustadas do GPT-2, com diferentes tamanhos e capacidades.

<h3 id="05.03">♟️ Estratégias de Decodificação para Controlar Aleatoriedade</h3> 
Durante a geração de texto, o modelo pode seguir diferentes estratégias para balancear criatividade e coerência. As principais técnicas abordadas são:
* Temperature Scaling: altera a distribuição de probabilidade dos próximos tokens, tornando a geração mais ou menos imprevisível.
* Top-k Sampling: limita a escolha dos próximos tokens a um conjunto com os k mais prováveis, evitando escolhas muito aleatórias.

Essas técnicas permitem personalizar a geração conforme o objetivo, seja ele mais criativo ou mais conservador.

<h3 id="05.04">🏗️ Carregamento de Pesos Pré-Treinados da OpenAI</h3> 
Por fim, é demonstrado como carregar modelos já treinados pela OpenAI, como o GPT-2, diretamente no modelo implementado. Isso permite aproveitar redes neurais treinadas em grandes conjuntos de dados, sem necessidade de treinar do zero, economizando tempo e recursos computacionais.

<div style="text-align: right;">
  <a href="#">🔝 Voltar ao topo</a>
</div>


<h1 id="06">✉️ 06 - Spam</h1>
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

<div style="text-align: right;">
  <a href="#">🔝 Voltar ao topo</a>
</div>

<h1 id="07">💪 07 - Intruções para Treinamento</h1>
Este projeto realiza o fine-tuning de um modelo GPT-2 (ex: gpt2-small, gpt2-medium, etc.) para a tarefa de pergunta e resposta baseada em instruções (Instruction Tuning).

Ao contrário do treinamento tradicional que se baseia apenas na predição da próxima palavra em sequências de texto, aqui utilizei um formato estruturado com instruções explícitas. Esse formato guia o modelo a entender e seguir comandos, tornando-o mais eficaz para tarefas práticas como assistentes virtuais, automação de suporte, geração de conteúdo e muito mais.

* O processo de ajuste fino de instruções adapta um LLM pré-treinado para seguir instruções humanas e gerar as respostas desejadas.
* A preparação do conjunto de dados envolve o download de um conjunto de dados de instruções-respostas, a formatação das entradas e a divisão em conjuntos de treinamento, validação e teste.
* Os lotes de treinamento são construídos usando uma função de agrupamento personalizada que preenche sequências, cria IDs de tokens de destino e mascara tokens de preenchimento.
* Carregamos um modelo GPT-2 pré-treinado com 124 milhões de parâmetros para servir como ponto de partida para o ajuste fino das instruções.
* O modelo pré-treinado é ajustado no conjunto de dados de instruções usando um loop de treinamento semelhante ao pré-treinamento.
* A avaliação envolve a extração das respostas do modelo em um conjunto de teste e sua pontuação (por exemplo, usando outro LLM).


<div style="text-align: right;">
  <a href="#">🔝 Voltar ao topo</a>
</div>

<h1 id="08">Referência</h1>
Build a Large Language Model - Sebastian Raschka<br>
Livros gratuitos para download: [Projeto Gutenberg](https://www.gutenberg.org/browse/languages/pt)<br>
Conjunto de dados utilizado para treinamento com pergunta e respostas: [Alpaca dataset - Stanford](https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json)


<div style="text-align: right;">
  <a href="#">🔝 Voltar ao topo</a>
</div>
