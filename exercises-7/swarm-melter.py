import numpy as np
import pyswarms as ps
import matplotlib.pyplot as plt
from pyswarms.utils.plotters import plot_cost_history


def endurance(x, y, z, u, v, w):
    return np.exp(-2 * (y - np.sin(x)) ** 2) + np.sin(z * u) + np.cos(v * w)


class MelterPSO:
    def __init__(self):
        self.dimensions = 6

        self.x_min = np.zeros(self.dimensions)
        self.x_max = np.ones(self.dimensions)
        self.bounds = (self.x_min, self.x_max)

        self.options = {
            "c1": 0.5,
            "c2": 0.3,
            "w": 0.9
        }

    def swarm_objective(self, swarm):
        costs = []
        for particle in swarm:
            x, y, z, u, v, w = particle
            costs.append(-endurance(x, y, z, u, v, w))
        return np.array(costs)


    def create_optimizer(self):
        return ps.single.GlobalBestPSO(
            n_particles=10,
            dimensions=self.dimensions,
            options=self.options,
            bounds=self.bounds
        )


    def run(self, iterations=100):
        optimizer = self.create_optimizer()

        best_cost, best_position = optimizer.optimize(
            self.swarm_objective,
            iters=iterations
        )

        self.cost_history = optimizer.cost_history

        print("\nRESULT")
        print("Best cost:", best_cost)
        print("Best position:", best_position)
        print("Real endurance:", -best_cost)

        return optimizer, best_position, best_cost


    def plot_cost(self):
        plot_cost_history(self.cost_history)
        plt.title("PSO Cost History")
        plt.show()
        plt.savefig("swarm-melter.png")


if __name__ == "__main__":
    pso = MelterPSO()

    optimizer, best_pos, best_cost = pso.run(iterations=100)

    pso.plot_cost()