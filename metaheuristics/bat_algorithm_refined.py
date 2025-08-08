import numpy as np


class BatAlgorithm:
    def __init__(self, fitness_function, dim, n_bats=100, max_iter=1000, A=0.5, r=0.5, alpha=0.9, gamma=0.9,
                 lower_bound=-1, upper_bound=1, save_history=True, elitism=True):
        self.fitness_function = fitness_function
        self.dim = dim
        self.n_bats = min(n_bats, 100)  # limite do enunciado
        self.max_iter = min(max_iter, 1000)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.A = np.full(self.n_bats, A)
        self.r = np.full(self.n_bats, r)
        self.alpha = alpha
        self.gamma = gamma

        # Frequência, velocidade e posição
        self.Q = np.zeros(self.n_bats)
        self.v = np.zeros((self.n_bats, dim))
        self.positions = np.random.uniform(lower_bound, upper_bound, (self.n_bats, dim))
        self.fitness = np.array([fitness_function(p) for p in self.positions])

        self.best_idx = np.argmin(self.fitness)
        self.best = self.positions[self.best_idx].copy()
        self.f_best = self.fitness[self.best_idx]

        self.save_history = save_history
        self.history = [] if save_history else None
        self.elitism = elitism

        # Debug: testar fitness com pesos aleatórios
        print("[DEBUG] Testando função de fitness com 5 vetores aleatórios:")
        for i in range(5):
            rand_weights = np.random.uniform(self.lower_bound, self.upper_bound, self.dim)
            fit = self.fitness_function(rand_weights)
            print(f"  Fitness aleatório {i+1}: {fit}")
        print("[DEBUG] Fitness dos primeiros morcegos da população:")
        for i in range(min(5, self.n_bats)):
            print(f"  Morcego {i}: fitness = {self.fitness[i]}")
        print(f"[DEBUG] Melhor fitness inicial: {self.f_best}")
        print(f"[DEBUG] Pesos do melhor morcego inicial (primeiros 10): {self.best[:10]}")
    def simple_bounds(self, sol):
        return np.clip(sol, self.lower_bound, self.upper_bound)

    def local_search(self, best, loudness):
        # Perturbação aleatória em torno do melhor
        epsilon = np.random.uniform(-1, 1, self.dim)
        return self.simple_bounds(best + epsilon * loudness)

    def run(self):
        for t in range(self.max_iter):
            # Elitismo: salva o melhor da geração anterior
            if self.elitism:
                elite_pos = self.best.copy()
                elite_fit = self.f_best

            for i in range(self.n_bats):
                beta = np.random.rand()
                self.Q[i] = beta
                self.v[i] += (self.positions[i] - self.best) * self.Q[i]
                new_position = self.positions[i] + self.v[i]
                new_position = self.simple_bounds(new_position)

                # Local search com probabilidade 1 - r[i]
                if np.random.rand() > self.r[i]:
                    new_position = self.local_search(self.best, self.A[i])

                f_new = self.fitness_function(new_position)

                # Critério de aceitação com A[i]
                if (f_new < self.fitness[i]) and (np.random.rand() < self.A[i]):
                    self.positions[i] = new_position
                    self.fitness[i] = f_new
                    self.A[i] *= self.alpha
                    self.r[i] = self.r[i] * (1 - np.exp(-self.gamma * t))

                    if f_new < self.f_best:
                        self.best = new_position
                        self.f_best = f_new

            # Elitismo: substitui o pior indivíduo pelo elite
            if self.elitism:
                worst_idx = np.argmax(self.fitness)
                self.positions[worst_idx] = elite_pos
                self.fitness[worst_idx] = elite_fit
                # Atualiza best/f_best caso o elitismo tenha restaurado um melhor
                self.best_idx = np.argmin(self.fitness)
                self.best = self.positions[self.best_idx].copy()
                self.f_best = self.fitness[self.best_idx]

            if self.save_history:
                self.history.append(self.f_best)

            print(f"Iteracao {t}: Melhor pontuação = {self.f_best:.2f}")
            if t % 10 == 0 or t == self.max_iter - 1:
                print(f"[DEBUG] Pesos do melhor agente (primeiros 10): {self.best[:10]}")
                print(f"[DEBUG] Fitness dos 5 primeiros morcegos: {[float(f) for f in self.fitness[:5]]}")

        if self.save_history:
            return self.best, self.f_best, self.history
        else:
            return self.best, self.f_best, None
