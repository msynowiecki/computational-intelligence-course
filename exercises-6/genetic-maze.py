import pygad
import numpy
import time
import matplotlib.pyplot as plt


class MazeGeneticAlgorithm:
    def __init__(self, maze, max_steps=30):
        self.maze = numpy.array(maze)
        self.rows, self.cols = self.maze.shape

        self.start = (0, 0)
        self.end = (self.rows - 1, self.cols - 1)

        self.max_steps = max_steps
        self.number_of_genes = max_steps

        self.gene_space = [
            {'low': 0.0, 'high': 1.0}
            for _ in range(self.number_of_genes)
        ]


    def can_move(self, x, y):
        return 0 <= x < self.rows and 0 <= y < self.cols and self.maze[x][y] == 0


    def decode_path(self, solution):
        x, y = self.start
        path = [(x, y)]

        for gene in solution:

            moves = []

            if self.can_move(x - 1, y):
                moves.append((x - 1, y))
            if self.can_move(x, y + 1):
                moves.append((x, y + 1))
            if self.can_move(x + 1, y):
                moves.append((x + 1, y))
            if self.can_move(x, y - 1):
                moves.append((x, y - 1))

            if len(moves) == 0:
                break

            index = int(gene * len(moves))
            x, y = moves[index]

            path.append((x, y))

            if (x, y) == self.end:
                break

        return path


    def fitness_function(self, ga_instance, solution, solution_index):

        path = self.decode_path(solution)

        x, y = path[-1]

        distance = abs(self.end[0] - x) + abs(self.end[1] - y)

        length_penalty = len(path)

        if (x, y) == self.end:
            return 100 - length_penalty

        return -distance - 0.1 * length_penalty


    def create_ga_instance(self):
        return pygad.GA(
            gene_space=self.gene_space,
            num_generations=300,
            num_parents_mating=20,
            fitness_func=self.fitness_function,
            sol_per_pop=60,
            num_genes=self.number_of_genes,
            parent_selection_type="tournament",
            keep_parents=5,
            crossover_type="single_point",
            mutation_type="random",
            mutation_percent_genes=15
        )


    def run_single_experiment(self):
        ga_instance = self.create_ga_instance()

        start = time.time()
        ga_instance.run()
        end = time.time()

        solution, fitness, _ = ga_instance.best_solution()

        return {
            "solution": solution,
            "fitness": fitness,
            "time": end - start,
            "ga_instance": ga_instance
        }


    def run_multiple_experiments(self, number_of_runs=10):

        best = None
        best_fitness = -1e9
        times = []

        for i in range(number_of_runs):
            result = self.run_single_experiment()
            path = self.decode_path(result["solution"])

            print(f"\nRUN {i+1}")
            print("Fitness:", result["fitness"])
            print("Time:", result["time"])
            print("Path length:", len(path))
            print("Path:", path)
            print("Genome:", result["solution"])

            times.append(result["time"])

            if result["fitness"] > best_fitness:
                best_fitness = result["fitness"]
                best = result

        print("\nSUMMARY")
        print("Best fitness:", best_fitness)
        print("Average time:", sum(times) / len(times))

        return best["ga_instance"]


    def display_best_solution(self, ga_instance):
        solution, fitness, _ = ga_instance.best_solution()
        path = self.decode_path(solution)

        print("\nBEST SOLUTION")
        print("Fitness:", fitness)
        print("Path length:", len(path))
        print("Path:", path)


    def save_fitness_plot(self, ga_instance, filename="genetic-maze.png"):
        ga_instance.plot_fitness()
        plt.savefig(filename)
        plt.close()
        print(f"Fitness plot saved to file: {filename}")


if __name__ == "__main__":

    maze = [
        [0,0,0,1,0,0,0,1,0,0],
        [1,1,0,0,0,1,0,1,1,0],
        [0,0,0,1,0,1,0,0,0,0],
        [0,1,0,1,1,0,0,1,1,0],
        [0,0,1,1,0,0,0,1,0,0],
        [0,0,0,0,0,1,0,0,0,1],
        [0,1,0,0,1,1,0,1,0,0],
        [0,1,0,0,1,1,0,1,0,0],
        [0,1,1,1,0,0,0,1,1,0],
        [0,1,0,1,1,0,1,0,1,0],
        [0,1,0,0,0,0,0,0,0,0]
    ]

    ga = MazeGeneticAlgorithm(maze, max_steps=30)

    ga_instance = ga.run_multiple_experiments(number_of_runs=10)

    ga.save_fitness_plot(ga_instance)

    ga.display_best_solution(ga_instance)




