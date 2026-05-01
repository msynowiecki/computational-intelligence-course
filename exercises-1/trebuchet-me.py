import random
import math
import matplotlib.pyplot as plt


def fire_distance(height, velocity, gravity, angle):
    radians = math.radians(angle)

    velocity_x = velocity * math.cos(radians)
    velocity_y = velocity * math.sin(radians)

    total_time = (velocity_y + math.sqrt(velocity_y ** 2 + 2 * gravity * height)) / gravity
    distance = velocity_x * total_time

    points = 100
    x = []
    y = []

    for i in range(points + 1):
        t = i * total_time / points
        x.append(velocity_x * t)
        y.append(height + velocity_y * t - 0.5 * gravity * t ** 2)

    return distance, x, y


def target_distance(min_distance, max_distance, offset):
    distance = random.randint(min_distance, max_distance)
    return distance, distance - offset, distance + offset


def correct_input(predicates, message = "Please input a value: ", error = "Invalid value!"):
    while True:
        value = input(message)

        if check_predicates(predicates, value):
            return value

        else:
            print(error)


def angle_input(message, error):
    angle = correct_input(
        [
            lambda value: isinstance(value, str),
            lambda value: value.isdigit(),
            lambda value: int(value) < 90,
            lambda value: int(value) > 0,
        ],
        message,
        error
    )
    return int(angle)


def fire_loop(height, velocity, gravity, min_distance, max_distance):
    while True:
        angle = angle_input(
            "Please input an angle (in degrees): ",
            "Invalid angle!",
        )

        shot_distance, x, y = fire_distance(height, velocity, gravity, angle)

        if min_distance <= shot_distance <= max_distance:

            plt.plot(x, y)
            plt.title("Trajectory")
            plt.xlabel("Distance [m]")
            plt.ylabel("Height [m]")
            plt.grid(True)
            plt.savefig("trebuchet-me.png")

            return
        elif shot_distance < min_distance:
            print("Too low!")
        elif shot_distance > max_distance:
            print("Too high!")


def check_predicates(predicates, value):
    for predicate in predicates:
        if not predicate(value):
            return False

    return True



def main():
    trebuchet_height = 100
    bullet_velocity = 50
    gravitational_constant = 9.81

    distance, min_distance, max_distance = target_distance(50, 340, 5)
    print(f"Target distance: {distance}")

    fire_loop(trebuchet_height, bullet_velocity, gravitational_constant, min_distance, max_distance)
    print("Target hit!")


main()