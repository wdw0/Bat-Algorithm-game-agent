import numpy as np
from game.core import SurvivalGame
from game.config import GameConfig
from agents.neural_network_agent import NeuralNetworkAgent

def main():
    # Carrega os pesos otimizados
    weights = np.load("best_agent_weights.npy")

    # Cria a configuração do jogo com render ativado
    config = GameConfig(num_players=1, render_grid=True)
    game = SurvivalGame(config, render=True)

    # Instancia o agente com a rede neural treinada
    agent = NeuralNetworkAgent(weights)

    # Loop principal do jogo
    while not game.all_players_dead():
        state = game.get_state(0, include_internals=True)
        action = agent.predict(state)
        game.update([action])
        game.render_frame()

    print(f"Pontuação final do agente: {game.players[0].score}")

if __name__ == "__main__":
    main()
