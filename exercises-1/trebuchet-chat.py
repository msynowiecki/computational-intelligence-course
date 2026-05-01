import math
import random
import matplotlib.pyplot as plt
import numpy as np

# Stałe
v0 = 50
h = 100
g = 9.81

# Losowanie celu
target = random.randint(50, 340)
print("Cel znajduje się w odległości:", target, "metrów")

attempts = 0

while True:
    attempts += 1

    angle_deg = float(input("Podaj kąt strzału (w stopniach): "))
    angle = math.radians(angle_deg)

    # Obliczenie czasu lotu
    t = (v0 * math.sin(angle) +
         math.sqrt((v0 * math.sin(angle)) ** 2 + 2 * g * h)) / g

    # Zasięg
    R = v0 * math.cos(angle) * t

    print("Pocisk spadł w odległości:", round(R, 2), "metrów")

    if target - 5 <= R <= target + 5:
        print("Cel trafiony!")
        print("Liczba prób:", attempts)
        break
    else:
        print("Nie trafiono, spróbuj ponownie.\n")

# =============================
# RYSOWANIE TRAJEKTORII
# =============================

# Generowanie czasu do momentu trafienia w ziemię
t_values = np.linspace(0, t, 500)

x = v0 * np.cos(angle) * t_values
y = h + v0 * np.sin(angle) * t_values - 0.5 * g * t_values ** 2

plt.figure()
plt.plot(x, y)
plt.grid()
plt.xlabel("Odległość (m)")
plt.ylabel("Wysokość (m)")
plt.title("Trajektoria pocisku Warwolf")

plt.savefig("trebuchet-chat.png")
plt.show()