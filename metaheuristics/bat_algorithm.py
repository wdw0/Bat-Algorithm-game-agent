import numpy as np

class BatAlgorithm:
    def __init__(self, fitness_function, dimension, population_size=100, generations=1000,
                 f_min=0.5, f_max=2.5, alpha=0.8, gamma=0.95, loudness_init=1, pulse_rate_init=0.5):
        self.fitness_function = fitness_function
        self.dimension = dimension
        self.population_size = population_size
        self.generations = generations

        self.f_min = f_min
        self.f_max = f_max
        self.alpha = alpha
        self.gamma = gamma

        self.A = np.ones(population_size) * loudness_init
        self.r = np.ones(population_size) * pulse_rate_init

        self.positions = np.random.uniform(-1, 1, (population_size, dimension))
        self.velocities = np.zeros((population_size, dimension))
        self.frequencies = np.zeros(population_size)

        self.fitness = np.array([fitness_function(p) for p in self.positions])
        self.best_index = np.argmax(self.fitness)
        self.best = self.positions[self.best_index].copy()
        self.best_score = self.fitness[self.best_index]

    def optimize(self, verbose=False):
        history = []

        for t in range(self.generations):
            for i in range(self.population_size):
                beta = np.random.rand()
                self.frequencies[i] = self.f_min + (self.f_max - self.f_min) * beta
                self.velocities[i] += (self.positions[i] - self.best) * self.frequencies[i]
                new_position = self.positions[i] + self.velocities[i]

                if np.random.rand() > self.r[i]:
                    epsilon = np.random.uniform(-1, 1, self.dimension)
                    new_position = self.best + epsilon * np.mean(self.A)

                new_score = self.fitness_function(new_position)

                if (np.random.rand() < self.A[i]) and (new_score > self.fitness[i]):
                    self.positions[i] = new_position
                    self.fitness[i] = new_score
                    self.A[i] *= self.alpha
                    self.r[i] = self.r[i] * (1 - np.exp(-self.gamma * t))

                if new_score > self.best_score:
                    self.best = new_position.copy()
                    self.best_score = new_score

            history.append(self.best_score)
            if verbose and t % 10 == 0:
                print(f"Iteração {t}: Melhor pontuação = {self.best_score:.2f}")

        return self.best, self.best_score, history
