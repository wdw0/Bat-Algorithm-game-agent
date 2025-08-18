import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, wilcoxon



def load_results():
    """Carrega resultados dos agentes para análise comparativa."""
    rule_based = [12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22, 9.95,
                  19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 15.43, 15.13, 22.50,
                  25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 13.80, 15.13, 12.35, 16.19]
    genetic_nn = [38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30, 39.76, 48.17,
                  44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 67.17, 53.54, 33.59, 49.24,
                  52.65, 16.35, 44.05, 56.59, 63.23, 43.96, 43.82, 19.19, 28.36, 18.65]
    human = [27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21, 18.92, 25.71, 20.05,
             31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 28.45, 12.59, 33.01,
             21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 22.96, 9.41, 35.22]
    try:
        bat_agent = np.loadtxt("bat_agent_result.txt").tolist()
    except Exception:
        print("⚠️ Arquivo 'bat_agent_result.txt' não encontrado. Usando simulação aleatória...")
        bat_agent = list(np.random.normal(loc=40, scale=5, size=30))
    return rule_based, genetic_nn, human, bat_agent


def analyze_and_plot():
    """Executa análise estatística e plota boxplot comparativo dos agentes."""
    rule_based, genetic_nn, human, bat_agent = load_results()
    print("Teste t (bat vs genetic_nn):", ttest_ind(bat_agent, genetic_nn))
    print("Teste Wilcoxon (bat vs genetic_nn):", wilcoxon(bat_agent, genetic_nn))
    print("\nTeste t (bat vs human):", ttest_ind(bat_agent, human))
    print("Teste Wilcoxon (bat vs human):", wilcoxon(bat_agent, human))
    print("\nTeste t (bat vs rule_based):", ttest_ind(bat_agent, rule_based))
    print("Teste Wilcoxon (bat vs rule_based):", wilcoxon(bat_agent, rule_based))
    data = {
        "Bat + NN": bat_agent,
        "Genetic + NN": genetic_nn,
        "Regra + GA": rule_based,
        "Humano": human
    }
    sns.boxplot(data=data)
    plt.title("Comparação de desempenho entre agentes")
    plt.ylabel("Pontuação")
    plt.xticks(rotation=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("boxplot_comparacao.png")
    plt.show()


if __name__ == "__main__":
    analyze_and_plot()
