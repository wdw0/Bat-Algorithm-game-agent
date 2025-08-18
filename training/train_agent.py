
import numpy as np
import matplotlib.pyplot as plt
import time
from metaheuristics.bat_algorithm import BatAlgorithm




    # Arquitetura compatível com best_agent_weights.npy
    input_size = 27
    hidden1 = 32
    hidden2 = 16
    output_size = 3
    dimension = (
        (input_size + 1) * hidden1 +
        (hidden1 + 1) * hidden2 +
        (hidden2 + 1) * output_size
    )

    def fitness_centered(w):
        from game.core import SurvivalGame
        from game.config import GameConfig
        from agents.neural_network_agent import NeuralNetworkAgent
        scores = []
        distances = []
        n_episodes = 10
        for _ in range(n_episodes):
            config = GameConfig(num_players=1, render_grid=False)
            game = SurvivalGame(config, render=False)
            agent = NeuralNetworkAgent(w)
            episode_distances = []
            while not game.all_players_dead():
                state = game.get_state(0, include_internals=True)
                action = agent.predict(state)
                game.update([action])
                center_y = config.screen_height / 2
                dist = abs(game.players[0].y - center_y)
                episode_distances.append(dist)
            scores.append(game.players[0].score)
            distances.append(np.mean(episode_distances))
        lambda_penalty = 0.01
        median_score = np.median(scores)
        mean_distance = np.mean(distances)
        return median_score - lambda_penalty * mean_distance

    bat = BatAlgorithm(
        fitness_function=fitness_centered,
        dimension=dimension,
        population_size=100,
        generations=1000,
        f_min=1.0,
        f_max=3.0,
        alpha=0.97,
        gamma=0.97,
        loudness_init=1.0,
        pulse_rate_init=0.5
    )

    start_time = time.time()
    half = bat.population_size // 2
    bat.positions[:half] = np.random.uniform(-5, 5, (half, dimension))
    bat.positions[half:] = np.random.uniform(-10, 10, (bat.population_size-half, dimension))
    bat.fitness = np.array([bat.fitness_function(p) for p in bat.positions])
    bat.best_index = np.argmax(bat.fitness)
    bat.best = bat.positions[bat.best_index].copy()
    bat.best_score = bat.fitness[bat.best_index]

    best_weights, best_score, history = bat.optimize(verbose=True)
    np.save("best_agent_weights.npy", best_weights)

    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_hours = elapsed // 3600
    elapsed_minutes = (elapsed % 3600) // 60
    elapsed_seconds = int(elapsed % 60)
    print(f"\nMelhor pontuação média encontrada: {best_score:.2f}")
    print(f"\n⏱️ Tempo total de execução: {int(elapsed_hours)}h {int(elapsed_minutes)}m {elapsed_seconds}s")
    plt.plot(history)
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Score (Mediana)")
    plt.title("Evolução do Agente - Bat Algorithm")
    plt.grid()
    plt.savefig("fitness_evolution.png")
    plt.show()


if __name__ == "__main__":
    main()
