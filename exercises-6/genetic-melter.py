import pygad
import time
import math
import matplotlib.pyplot as plt


def endurance(x, y, z, u, v, w):
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


class MelterGeneticAlgorithm:
    def __init__(self, optimal_value=None):
        self.optimal_value = optimal_value
        self.number_of_genes = 6
        self.gene_space = [{'low': 0.0, 'high': 1.0} for _ in range(self.number_of_genes)]

    def fitness_function(self, ga_instance, solution, solution_index):
        x, y, z, u, v, w = solution

        return endurance(x, y, z, u, v, w)

    def create_ga_instance(self):
        return pygad.GA(
            gene_space=self.gene_space,
            num_generations=160,
            num_parents_mating=10,
            fitness_func=self.fitness_function,
            sol_per_pop=20,
            num_genes=self.number_of_genes,
            parent_selection_type="sss",
            keep_parents=2,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=20,
            random_seed=None
        )

    def run_single_experiment(self):
        ga_instance = self.create_ga_instance()

        start_time = time.time()
        ga_instance.run()
        end_time = time.time()

        solution, fitness, _ = ga_instance.best_solution()

        return {
            "solution": solution,
            "fitness": fitness,
            "execution_time": end_time - start_time,
            "ga_instance": ga_instance
        }

    def run_multiple_experiments(self, number_of_runs=10):
        best_ga_instance = None

        best_solution = None
        best_fitness = -float("inf")

        for run_index in range(number_of_runs):
            result = self.run_single_experiment()

            print(f"\nRUN {run_index + 1}")
            print("Solution:", result["solution"])
            print("Endurance:", result["fitness"])
            print("Time:", result["execution_time"])
            print("Generations:", result["ga_instance"].generations_completed)

            if result["fitness"] > best_fitness:
                best_fitness = result["fitness"]
                best_solution = result["solution"]
                best_ga_instance = result["ga_instance"]

        print("\nSUMMARY")
        print("Best endurance found:", best_fitness)
        print("Best alloy proportions:", best_solution)

        return best_ga_instance

    def save_fitness_plot(self, ga_instance, filename="genetic-melter.png"):
        ga_instance.plot_fitness()
        plt.savefig(filename)
        plt.close()
        print(f"Saved plot: {filename}")

    def display_best_solution(self, ga_instance):
        best_solution, best_fitness, _ = ga_instance.best_solution()

        print("\nBEST SOLUTION")
        print("Proportions:", best_solution)
        print("Endurance:", best_fitness)


if __name__ == "__main__":
    ga = MelterGeneticAlgorithm()

    ga_instance = ga.run_multiple_experiments(number_of_runs=10)

    ga.save_fitness_plot(ga_instance)

    ga.display_best_solution(ga_instance)