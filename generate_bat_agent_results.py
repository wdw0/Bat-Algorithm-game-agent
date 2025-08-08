import numpy as np
from utils import evaluate

def main():
    weights = np.load("best_agent_weights.npy")
    scores = []

    for i in range(30):
        score = evaluate(weights, n_episodes=1)
        scores.append(score)
        print(f"Execução {i+1}/30: {score:.2f}")

    # Salva em formato compatível com o script de avaliação
    np.savetxt("bat_agent_result.txt", scores, fmt='%.2f')
    mean = np.mean(scores)
    std = np.std(scores)
    # Salva média e desvio padrão ao final do arquivo
    with open("bat_agent_result.txt", "a") as f:
        f.write(f"\nMédia: {mean:.2f}\n")
        f.write(f"Desvio padrão: {std:.2f}\n")
    print("\nResultados salvos em 'bat_agent_result.txt'.")
    print(f"Média dos scores: {mean:.2f}")
    print(f"Desvio padrão dos scores: {std:.2f}")

if __name__ == "__main__":
    main()
