import numpy as np
import matplotlib.pyplot as plt
import time
from utils import evaluate
#from metaheuristics.bat_algorithm_refined import BatAlgorithm
from metaheuristics.bat_algorithm import BatAlgorithm


def main():
    # Número de pesos da rede (27->32->16->3 com bias):
    dimension = 1475  # calculado previamente

    # Inicializa o otimizador
    """    bat = BatAlgorithm(
        fitness_function=lambda w: evaluate(w, n_episodes=2),
        dimension=dimension,
        population_size=100,
        generations=1000,
        f_min=0.5,
        f_max=2.5,
        alpha=0.8,
        gamma=0.95,
        loudness_init=1.0,
        pulse_rate_init=0.5
    )
    """
    bat = BatAlgorithm(
        fitness_function=lambda w: evaluate(w, n_episodes=5),
        dimension=dimension,
        population_size=100,         # ← máximo permitido
        generations=1000,            # ← máximo permitido
        f_min=0.5,
        f_max=2.5,
        alpha=0.95,
        gamma=0.9,
        loudness_init=0.9,
        pulse_rate_init=0.1
    )

    
    start_time = time.time()
    # Roda a otimização
    
    best_weights, best_score, history = bat.optimize(verbose=True)
    #best_weights, best_score, history = bat.run()

    # Salva os melhores pesos
    np.save("best_agent_weights.npy", best_weights)
     
    end_time = time.time()
    
    elapsed = end_time - start_time
    elapsed_hours = elapsed // 3600
    elapsed_minutes = (elapsed % 3600) // 60
    elapsed_seconds = int(elapsed % 60)


    print(f"\nMelhor pontuação média encontrada: {best_score:.2f}")
    print(f"\n⏱️ Tempo total de execução: {int(elapsed_hours)}h {int(elapsed_minutes)}m {elapsed_seconds}s")

    # Plota a evolução do fitness ao longo das iterações
    plt.plot(history)
    plt.xlabel("Iteração")
    plt.ylabel("Melhor Score Médio")
    plt.title("Evolução do Agente - Bat Algorithm")
    plt.grid()
    plt.savefig("fitness_evolution.png")
    plt.show()



if __name__ == "__main__":
    main()
