import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size=27, hidden1=32, hidden2=16, output_size=3):
        """
        Inicializa a arquitetura da rede neural com duas camadas ocultas e uma de saída (compatível com best_agent_weights.npy antigo).
        """
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size

        # Total de parâmetros (pesos e bias)
        self.total_weights = (
            (input_size + 1) * hidden1 +     # pesos + bias da camada 1
            (hidden1 + 1) * hidden2 +       # pesos + bias da camada 2
            (hidden2 + 1) * output_size     # pesos + bias da camada de saída
        )

        self.set_weights(np.zeros(self.total_weights))

    def set_weights(self, weights: np.ndarray):
        """
        Atribui os pesos da rede neural a partir de um vetor linear.
        Esse vetor será fornecido pela metaheurística (ex: Bat Algorithm).
        """
        idx = 0

        # Camada 1: entrada -> hidden1
        w1_size = (self.input_size + 1) * self.hidden1  # +1 por conta do bias
        self.w1 = weights[idx:idx + w1_size].reshape((self.input_size + 1, self.hidden1))
        idx += w1_size

        # Camada 2: hidden1 -> hidden2
        w2_size = (self.hidden1 + 1) * self.hidden2
        self.w2 = weights[idx:idx + w2_size].reshape((self.hidden1 + 1, self.hidden2))
        idx += w2_size

        # Camada de saída: hidden2 -> output
        w3_size = (self.hidden2 + 1) * self.output_size
        self.w3 = weights[idx:idx + w3_size].reshape((self.hidden2 + 1, self.output_size))

    def forward(self, x: np.ndarray) -> int:
        """
        Executa a propagação direta da rede e retorna a ação prevista (0, 1 ou 2).
        """
        # Entrada com bias
        x = np.append(x, 1)
        h1 = np.tanh(np.dot(x, self.w1))
        h1 = np.append(h1, 1)
        h2 = np.tanh(np.dot(h1, self.w2))
        h2 = np.append(h2, 1)
        out = self._softmax(np.dot(h2, self.w3))
        return int(np.argmax(out))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Função de ativação softmax para a camada de saída.
        Retorna uma distribuição de probabilidade entre as 3 ações possíveis.
        """
        exps = np.exp(x - np.max(x))  # para estabilidade numérica
        return exps / np.sum(exps)

    def raw_output(self, state: np.ndarray) -> np.ndarray:
        """
        Retorna as ativações da camada de saída sem aplicar softmax.
        Útil para depuração e análise.
        """
        x = np.append(state, 1)
        h1 = np.tanh(np.dot(x, self.w1))
        h1 = np.append(h1, 1)
        h2 = np.tanh(np.dot(h1, self.w2))
        h2 = np.append(h2, 1)
import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size=27, hidden1=32, hidden2=16, output_size=3):
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.output_size = output_size
        self.total_weights = (
            (input_size + 1) * hidden1 +
            (hidden1 + 1) * hidden2 +
            (hidden2 + 1) * output_size
        )
        self.set_weights(np.zeros(self.total_weights))

    def set_weights(self, weights: np.ndarray):
        idx = 0
        w1_size = (self.input_size + 1) * self.hidden1
        self.w1 = weights[idx:idx + w1_size].reshape((self.input_size + 1, self.hidden1))
        idx += w1_size
        w2_size = (self.hidden1 + 1) * self.hidden2
        self.w2 = weights[idx:idx + w2_size].reshape((self.hidden1 + 1, self.hidden2))
        idx += w2_size
        w3_size = (self.hidden2 + 1) * self.output_size
        self.w3 = weights[idx:idx + w3_size].reshape((self.hidden2 + 1, self.output_size))

    def forward(self, x: np.ndarray) -> int:
        x = np.append(x, 1)
        h1 = np.tanh(np.dot(x, self.w1))
        h1 = np.append(h1, 1)
        h2 = np.tanh(np.dot(h1, self.w2))
        h2 = np.append(h2, 1)
        out = self._softmax(np.dot(h2, self.w3))
        return int(np.argmax(out))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps)

    def raw_output(self, state: np.ndarray) -> np.ndarray:
        x = np.append(state, 1)
        h1 = np.tanh(np.dot(x, self.w1))
        h1 = np.append(h1, 1)
        h2 = np.tanh(np.dot(h1, self.w2))
        h2 = np.append(h2, 1)
        return np.dot(h2, self.w3)
