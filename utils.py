import numpy as np
from game.core import SurvivalGame
from game.config import GameConfig
from agents.neural_network_agent import NeuralNetworkAgent


def evaluate(weights: np.ndarray, n_episodes: int = 5, render: bool = True) -> float:
    """
    Avalia o desempenho de um conjunto de pesos em múltiplas execuções do jogo.

    :param weights: vetor 1D com os pesos da rede neural
    :param n_episodes: número de partidas para rodar (default = 3)
    :param render: se True, exibe o jogo visualmente (usar apenas para debug)
    :return: média das pontuações obtidas nas partidas
    """
    scores = []

    for episode in range(n_episodes):
        # Cria a configuração do jogo para um único agente
        config = GameConfig(num_players=1, render_grid=False)
        game = SurvivalGame(config, render=render)

        # Instancia o agente com os pesos fornecidos
        agent = NeuralNetworkAgent(weights)

        # Loop do jogo
        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            if render:
                game.render_frame()
        # Armazena o score final deste episódio
        scores.append(game.players[0].score)

    # Retorna a média das pontuações como fitness
    return float(np.mean(scores))
