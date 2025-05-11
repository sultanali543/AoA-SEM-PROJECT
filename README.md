# AoA-SEM-PROJECT
# THIS REPOSITORY CONTAIN DETAILS AND CODE IMPLIMENTATION OF AOA SEMESTER PROJECT

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


data = load_breast_cancer()
X, y = data.data, data.target
n_features = X.shape[1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def fitness_function(solution):
    if np.sum(solution) == 0:
        return 1e5

    selected_indices = np.where(solution == 1)[0]
    X_sel_train = X_train[:, selected_indices]
    X_sel_test = X_test[:, selected_indices]

    clf = LogisticRegression(max_iter=500, solver='liblinear')
    clf.fit(X_sel_train, y_train)
    accuracy = clf.score(X_sel_test, y_test)

    feature_ratio = np.sum(solution) / len(solution)
    return (1 - accuracy) + 0.01 * feature_ratio


def initialize_population(pop_size, dim):
    return np.random.randint(0, 2, size=(pop_size, dim))

def update_energy(E0, t, T):
    return 2 * E0 * (1 - t / T)

def update_position(vulture, best, E):
    r1, r2 = np.random.rand(), np.random.rand()
    if abs(E) >= 1:
        new_pos = np.where(np.random.rand(len(vulture)) < 0.5, best, 1 - vulture)
    else:
        new_pos = np.where(np.random.rand(len(vulture)) < r1, best, vulture)
    return np.clip(new_pos.round(), 0, 1)


def AVO_FS(pop_size=10, max_iter=15):
    population = initialize_population(pop_size, n_features)
    fitness = np.array([fitness_function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    fitness_curve = [best_fitness]

    for t in range(max_iter):
        E0 = np.random.uniform(-1, 1)
        E = update_energy(E0, t, max_iter)

        for i in range(pop_size):
            population[i] = update_position(population[i], best_solution, E)

        fitness = np.array([fitness_function(ind) for ind in population])
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_solution = population[current_best_idx].copy()

        fitness_curve.append(best_fitness)
        print(f"Iter {t+1}: Fitness = {best_fitness:.4f}, Features Selected = {np.sum(best_solution)}")

    
    plt.plot(fitness_curve, marker='o')
    plt.title("AVO Feature Selection - Optimized Version")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness (Lower is Better)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_solution, best_fitness


if __name__ == "__main__":
    best_features, best_score = AVO_FS()
    print("\nBest Feature Subset (1 = selected):\n", best_features)
    print("Final Fitness Score:", best_score)
    print("Total Selected Features:", np.sum(best_features))

