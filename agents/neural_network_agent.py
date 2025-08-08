import numpy as np
from game.agents import Agent  # Interface exigida pelo jogo
from agents.simple_nn import SimpleNeuralNetwork  # Sua rede neural manual

class NeuralNetworkAgent(Agent):
    """
    Agente inteligente que utiliza uma rede neural para prever ações
    com base no estado do jogo.
    """
    def __init__(self, weights: np.ndarray):
        """
        Inicializa o agente com um vetor de pesos que será usado para
        configurar os parâmetros da rede neural.

        :param weights: vetor 1D com os pesos otimizados para a rede
        """
        self.nn = SimpleNeuralNetwork()
        self.nn.set_weights(weights)  # Configura os pesos da rede

    def predict(self, state: np.ndarray) -> int:
        """
        Recebe o vetor de estado do jogo e retorna a ação a ser executada,
        evitando que o agente se mova para fora dos limites verticais.

        Ações:
        0 - noop (ficar parado)
        1 - mover para cima
        2 - mover para baixo
        """
        # Extrai a posição vertical do jogador do vetor de estado
        y_position = state[25]  # valor normalizado (0.0 a 1.0)


        grid_height = 12  # Conforme definido em config.py
        real_y = y_position * (grid_height - 1)

        # Obtém as ativações da rede neural (sem aplicar softmax nem argmax)
        outputs = self.nn.raw_output(state)

        if real_y < 1:
            outputs[1] = -np.inf  # não subir
        elif real_y >= grid_height - 1.5:
            outputs[2] = -np.inf  # não descer

        
            
        # Seleciona a melhor ação restante
        return int(np.argmax(outputs))

