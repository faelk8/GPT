import torch
from torch.utils.data import Dataset
import pandas as pd


class SpamDataset(Dataset):

    # Método de inicialização do dataset
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        """
        Inicializa o dataset a partir de um arquivo CSV.

        Parâmetros:
        - csv_file (str): Caminho para o arquivo CSV contendo as colunas 'Text' e 'Label'.
        - tokenizer (objeto): Um tokenizador que possui o método .encode() para tokenizar os textos.
        - max_length (int, opcional): Tamanho máximo dos textos. Se None, será calculado automaticamente.
        - pad_token_id (int, opcional): ID do token de preenchimento (padding) a ser usado. Padrão: 50256.

        A função:
        - Lê os dados do CSV.
        - Tokeniza os textos.
        - Trunca os textos para o tamanho máximo.
        - Aplica padding para uniformizar o comprimento.

        Não retorna nada. Os dados processados ficam armazenados como atributos da instância.
        """
        # Lê os dados do arquivo CSV para um DataFrame do pandas
        self.data = pd.read_csv(csv_file)

        # Aplica o tokenizador em cada texto da coluna "Text", transformando os textos em listas de IDs de tokens
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]
        ]

        # Se max_length não for informado, define como o comprimento do maior texto codificado
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length

        # Trunca os textos codificados se excederem o tamanho máximo
        self.encoded_texts = [
            encoded_text[:self.max_length]
            for encoded_text in self.encoded_texts
        ]

        # Preenche (padding) os textos codificados com o token de preenchimento até atingirem o tamanho máximo
        self.encoded_texts = [
            encoded_text + [pad_token_id] *
            (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    # Método que retorna um item do dataset com base no índice
    def __getitem__(self, index):
        """
        Retorna o exemplo do dataset no índice especificado.

        Parâmetros:
        - index (int): Índice do exemplo desejado.

        Retorna:
        - tuple: (input_tensor, label_tensor), ambos do tipo torch.Tensor e dtype=torch.long.
        """
        # Obtém o texto codificado no índice
        encoded = self.encoded_texts[index]
        # Obtém o rótulo (label) correspondente
        label = self.data.iloc[index]["Label"]
        return (
            # Retorna o texto como tensor do tipo long
            torch.tensor(encoded, dtype=torch.long),
            # Retorna o rótulo como tensor do tipo long
            torch.tensor(label, dtype=torch.long)
        )

    # Método que retorna o número total de exemplos no dataset
    def __len__(self):
        """
        Retorna o número total de exemplos no dataset.

        Retorna:
        - int: Quantidade de linhas (exemplos) no arquivo CSV.
        """
        return len(self.data)

    # Método auxiliar que encontra o comprimento do maior texto codificado
    def _longest_encoded_length(self):
        """
        Calcula o comprimento do maior texto tokenizado presente no dataset.

        Retorna:
        - int: O maior comprimento (número de tokens) entre todos os textos tokenizados.
        """
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length

        # Observação: também é possível reescrever esse método de forma mais concisa e pythonica:
        # return max(len(encoded_text) for encoded_text in self.encoded_texts)
