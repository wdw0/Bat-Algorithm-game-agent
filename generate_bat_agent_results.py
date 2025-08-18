def main():

import numpy as np
from utils import evaluate

def main():
    weights = np.load("best_agent_weights.npy")
    scores = [evaluate(weights, n_episodes=1, render=False) for _ in range(30)]
    for i, score in enumerate(scores):
        print(f"Execução {i+1}/30: {score:.2f}")
    np.savetxt("bat_agent_result.txt", scores, fmt='%.2f')
    mean = np.mean(scores)
    std = np.std(scores)
    with open("bat_agent_result.txt", "a") as f:
        f.write(f"\nMédia: {mean:.2f}\nDesvio padrão: {std:.2f}\n")
    print(f"\nResultados salvos em 'bat_agent_result.txt'.\nMédia dos scores: {mean:.2f}\nDesvio padrão dos scores: {std:.2f}")

if __name__ == "__main__":
    main()
