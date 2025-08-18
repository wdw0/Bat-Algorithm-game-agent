def evaluate(weights: np.ndarray, n_episodes: int = 5, render: bool = True) -> float:
import numpy as np
from game.core import SurvivalGame
from game.config import GameConfig
from agents.neural_network_agent import NeuralNetworkAgent

def evaluate(weights: np.ndarray, n_episodes: int = 5, render: bool = True) -> float:
    scores = []
    for _ in range(n_episodes):
        config = GameConfig(num_players=1, render_grid=False)
        game = SurvivalGame(config, render=render)
        agent = NeuralNetworkAgent(weights)
        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            if render:
                game.render_frame()
        scores.append(game.players[0].score)
    return float(np.mean(scores))
