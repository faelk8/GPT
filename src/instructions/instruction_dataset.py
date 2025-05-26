from torch.utils.data import Dataset
from instructions.instruction import format_input


class InstructionDataset(Dataset):
    """
    Dataset personalizado para tarefas de aprendizado supervisionado com modelos de linguagem.

    Cada item no dataset é uma sequência completa de tokens que representa:
    - Uma instrução formatada (e opcionalmente um input);
    - Uma resposta esperada.

    A sequência final é formada por:
    [instruction_plus_input] + "\n\n### Response:\n" + [output]

    Parâmetros:
    ----------
    data : List[Dict]
        Lista de dicionários contendo, por exemplo, as chaves 'instruction', 'input' e 'output'.

    tokenizer : transformers.PreTrainedTokenizer
        Tokenizador utilizado para converter os textos completos em IDs de tokens.

    Atributos:
    ----------
    encoded_texts : List[List[int]]
        Lista de listas contendo os IDs dos tokens de cada amostra, já tokenizadas.
    """

    def __init__(self, data, tokenizer):
        self.data = data  # Armazena a lista original de exemplos

        self.encoded_texts = []  # Lista onde serão armazenados os exemplos tokenizados

        for entry in data:
            # Gera o texto de entrada a partir do dicionário (instrução + input), usando função externa
            instruction_plus_input = format_input(entry)

            # Gera o texto da resposta, precedido por um separador padronizado
            response_text = f"\n\n### Response:\n{entry['output']}"

            # Concatena a instrução + input com a resposta
            full_text = instruction_plus_input + response_text

            # Tokeniza o texto completo e adiciona à lista
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        """
        Retorna a sequência de tokens para o exemplo no índice especificado.
        """
        return self.encoded_texts[index]

    def __len__(self):
        """
        Retorna o número total de exemplos no dataset.
        """
        return len(self.data)
