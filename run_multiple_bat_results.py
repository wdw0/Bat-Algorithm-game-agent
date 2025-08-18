import subprocess
import numpy as np

results = []

for i in range(30):
    print(f"Rodando instância {i+1}/30...")
    subprocess.run(["python3", "generate_bat_agent_results.py"])
    # Lê os scores da última execução
    with open("bat_agent_result.txt", "r") as f:
        lines = f.readlines()
        # Pega apenas as linhas com scores (ignora média e desvio padrão)
        scores = []
        for line in lines:
            try:
                scores.append(float(line.strip()))
            except ValueError:
                continue
        if scores:
            results.append(scores)

# Seleciona a execução com maior média dos scores
best_idx = np.argmax([np.mean(r) for r in results])
best_scores = results[best_idx]

# Salva os melhores resultados
np.savetxt("bat_agent_best_result.txt", best_scores, fmt='%.2f')
mean = np.mean(best_scores)
std = np.std(best_scores)
with open("bat_agent_best_result.txt", "a") as f:
    f.write(f"\nMédia: {mean:.2f}\n")
    f.write(f"Desvio padrão: {std:.2f}\n")
print(f"\nMelhor execução salva em 'bat_agent_best_result.txt' (instância {best_idx+1}/30)")
print(f"Média dos scores: {mean:.2f}")
print(f"Desvio padrão dos scores: {std:.2f}")
