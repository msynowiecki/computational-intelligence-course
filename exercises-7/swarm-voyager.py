import random
import time
from aco import AntColony


class MelterACO:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.best_path = None
        self.best_distance = None
        self.last_colony = None

    def run_single(self,
                   ant_count=150,
                   alpha=1,
                   beta=2,
                   evaporation=0.4,
                   pheromone_constant=1000,
                   iterations=200):

        colony = AntColony(
            self.coordinates,
            ant_count=ant_count,
            alpha=alpha,
            beta=beta,
            pheromone_evaporation_rate=evaporation,
            pheromone_constant=pheromone_constant,
            iterations=iterations
        )

        start = time.time()
        best_path = colony.get_path()
        end = time.time()

        self.last_colony = colony

        self.best_path = best_path

        print("\nRUN RESULT")
        print("Best path:", best_path)
        print("Execution time:", end - start)

        return best_path

    def run_multiple(self, runs=5):
        best_global = None

        print("\nMULTIPLE RUNS")

        for i in range(runs):
            print(f"\nRun {i+1}")

            result = self.run_single(
                ant_count=150,
                alpha=1,
                beta=2,
                evaporation=0.4,
                pheromone_constant=1000,
                iterations=200
            )

            best_global = result

        print("\nSUMMARY")
        print("Last best path:", best_global)

        return best_global


    def parameter_study(self):
        configs = [
            {"alpha": 0.5, "beta": 3, "evaporation": 0.3},
            {"alpha": 1, "beta": 2, "evaporation": 0.4},
            {"alpha": 2, "beta": 1, "evaporation": 0.6},
        ]

        results = []

        for index, config in enumerate(configs):
            print(f"\nCONFIG {index+1}")

            self.run_single(
                ant_count=200,
                alpha=config["alpha"],
                beta=config["beta"],
                evaporation=config["evaporation"],
                pheromone_constant=1200,
                iterations=250
            )

            results.append(config)

        return results


def generate_random_coordinates(n=15):
    return tuple(
        (random.randint(0, 100), random.randint(0, 100))
        for _ in range(n)
    )


if __name__ == "__main__":

    coordinates = generate_random_coordinates(15)

    model = MelterACO(coordinates)

    model.run_multiple(runs=3)

    model.parameter_study()

    grid_coordinates = tuple(
        (x, y)
        for y in range(0, 50, 10)
        for x in range(0, 50, 10)
    )

    grid_model = MelterACO(grid_coordinates)

    grid_model.run_single(
        ant_count=300,
        alpha=1,
        beta=2,
        evaporation=0.4,
        pheromone_constant=1200,
        iterations=300
    )