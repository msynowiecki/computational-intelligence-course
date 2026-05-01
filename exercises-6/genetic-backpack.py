import pygad
import numpy
import time
import matplotlib.pyplot as plt


class BackpackGeneticAlgorithm:
    def __init__(self, weights, values, max_weight, optimal_value=None):
        self.weights = numpy.array(weights)
        self.values = numpy.array(values)
        self.max_weight = max_weight
        self.optimal_value = optimal_value

        self.gene_space = [0, 1]
        self.number_of_genes = len(weights)

    def fitness_function(self, ga_instance, solution, solution_index):
        total_weight = numpy.sum(solution * self.weights)
        total_value = numpy.sum(solution * self.values)

        if total_weight > self.max_weight:
            penalty = 10 * (total_weight - self.max_weight)
            return total_value - penalty

        return total_value

    def create_ga_instance(self):
        return pygad.GA(
            gene_space=self.gene_space,
            num_generations=100,
            num_parents_mating=10,
            fitness_func=self.fitness_function,
            sol_per_pop=10,
            num_genes=self.number_of_genes,
            parent_selection_type="sss",
            keep_parents=2,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=8
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
        successful_runs = 0
        execution_times = []
        last_ga_instance = None

        for run_index in range(number_of_runs):
            result = self.run_single_experiment()
            last_ga_instance = result["ga_instance"]

            print(f"\nRUN {run_index + 1}")
            print("Solution:", result["solution"])
            print("Total value:", result["fitness"])
            print("Execution time:", result["execution_time"])
            print("Completed generations:", last_ga_instance.generations_completed)

            if self.optimal_value is not None and result["fitness"] == self.optimal_value:
                successful_runs += 1
                execution_times.append(result["execution_time"])

        print("\nSUMMARY")
        print("Success rate:", successful_runs * 100 / number_of_runs)

        if execution_times:
            print("Average execution time:", sum(execution_times) / len(execution_times))
        else:
            print("No successful runs recorded.")

        return last_ga_instance

    def display_best_solution(self, ga_instance):
        best_solution, best_fitness, _ = ga_instance.best_solution()

        selected_items = [index for index in range(len(best_solution)) if best_solution[index] == 1]
        total_weight = numpy.sum(self.weights * best_solution)

        print("\nBEST SOLUTION")
        print("Selected item indices:", selected_items)
        print("Total weight:", total_weight)
        print("Total value:", best_fitness)

    def save_fitness_plot(self, ga_instance, filename="genetic-backpack.png"):
        ga_instance.plot_fitness()
        plt.savefig(filename)
        plt.close()
        print(f"Fitness plot saved to file: {filename}")


if __name__ == "__main__":
    weights = [2, 5, 10, 5, 3, 8, 7, 4, 6, 9]
    values  = [100, 300, 500, 200, 150, 400, 350, 180, 220, 250]
    max_weight = 25

    OPTIMAL_VALUE = 1630

    ga = BackpackGeneticAlgorithm(weights, values, max_weight, OPTIMAL_VALUE)

    ga_instance = ga.run_multiple_experiments(number_of_runs=10)

    ga.save_fitness_plot(ga_instance)

    ga.display_best_solution(ga_instance)