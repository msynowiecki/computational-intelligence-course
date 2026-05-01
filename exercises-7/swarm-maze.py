import numpy as np
import random
import time


class MazeACO:
    def __init__(self, maze, start=(0, 0), end=None):
        self.maze = np.array(maze)
        self.rows, self.cols = self.maze.shape

        self.start = start
        self.end = end or (self.rows - 1, self.cols - 1)

        self.graph = self.build_graph()

        self.best_path = None
        self.best_length = float("inf")


    def build_graph(self):
        graph = {}

        for x in range(self.rows):
            for y in range(self.cols):
                if self.maze[x][y] == 0:
                    neighbors = []

                    for adjacent_x, adjacent_y in [(-1,0),(1,0),(0,-1),(0,1)]:
                        neighbor_x, neighbor_y = x + adjacent_x, y + adjacent_y

                        if 0 <= neighbor_x < self.rows and 0 <= neighbor_y < self.cols:
                            if self.maze[neighbor_x][neighbor_y] == 0:
                                neighbors.append((neighbor_x, neighbor_y))

                    graph[(x, y)] = neighbors

        return graph


    def build_path(self):
        current = self.start
        path = [current]

        visited = set()
        visited.add(current)

        for _ in range(self.rows * self.cols):

            neighbors = [
                neighbor for neighbor in self.graph.get(current, [])
                if neighbor not in visited
            ]

            if not neighbors:
                break

            current = random.choice(neighbors)
            path.append(current)
            visited.add(current)

            if current == self.end:
                break

        return path


    def path_cost(self, path):

        if path[-1] != self.end:
            return float("inf")

        return len(path)


    def run(self, ants=100, iterations=200):

        start_time = time.time()

        for iteration in range(iterations):

            all_paths = []

            for _ in range(ants):
                path = self.build_path()
                cost = self.path_cost(path)

                all_paths.append((path, cost))

                if cost < self.best_length:
                    self.best_length = cost
                    self.best_path = path

            print(f"Iteration {iteration+1} | best length: {self.best_length}")

        end_time = time.time()

        print("\nFINAL RESULT")
        print("Best path:", self.best_path)
        print("Length:", self.best_length)
        print("Time:", end_time - start_time)

        return self.best_path


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

    model = MazeACO(maze)

    best_path = model.run(
        ants=150,
        iterations=100
    )