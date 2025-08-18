import numpy as np
from game.agents import Agent
from agents.simple_nn import SimpleNeuralNetwork


class NeuralNetworkAgent(Agent):
    """
    Agente que utiliza uma rede neural para prever ações com base no estado do jogo.
    """
    def __init__(self, weights: np.ndarray):
        self.nn = SimpleNeuralNetwork()
        self.nn.set_weights(weights)

    def predict(self, state: np.ndarray) -> int:
        y_position = state[25]
        grid_height = 12
        real_y = y_position * (grid_height - 1)
        outputs = self.nn.raw_output(state)
        if real_y < 1:
            outputs[1] = -np.inf
        elif real_y >= grid_height - 1.5:
            outputs[2] = -np.inf
        return int(np.argmax(outputs))

