import math
from datetime import date

# Pobieranie danych od użytkownika
name = input("Podaj swoje imię: ")
year = int(input("Podaj rok urodzenia (rrrr): "))
month = int(input("Podaj miesiąc urodzenia (mm): "))
day = int(input("Podaj dzień urodzenia (dd): "))

# Obliczenie daty urodzenia i dzisiejszej daty
birth_date = date(year, month, day)
today = date.today()

# Obliczenie liczby dni życia
days_lived = (today - birth_date).days

print(f"\nWitaj {name}!")
print(f"Dzisiaj jest {days_lived}. dzień Twojego życia.\n")


# Funkcja licząca biorytm
def biorhythm(days, cycle_length):
    return math.sin(2 * math.pi * days / cycle_length)


# Obliczenia biorytmów
physical = biorhythm(days_lived, 23)
emotional = biorhythm(days_lived, 28)
intellectual = biorhythm(days_lived, 33)

# Obliczenia na jutro (potrzebne do porównania)
physical_tomorrow = biorhythm(days_lived + 1, 23)
emotional_tomorrow = biorhythm(days_lived + 1, 28)
intellectual_tomorrow = biorhythm(days_lived + 1, 33)


# Funkcja oceniająca wynik
def evaluate(name_cycle, value, value_tomorrow):
    print(f"{name_cycle}: {value:.2f}")

    if value > 0.5:
        print("Świetny wynik! To może być bardzo dobry dzień! 😊")
    elif value < -0.5:
        print("To może być trudniejszy dzień...")
        if value_tomorrow > value:
            print("Nie martw się. Jutro będzie lepiej! 🌅")
        else:
            print("Jutro może być jeszcze trudniej – zadbaj o siebie ❤️")
    else:
        print("Dzień raczej neutralny.")

    print()


# Wyświetlenie wyników
evaluate("Biorytm fizyczny", physical, physical_tomorrow)
evaluate("Biorytm emocjonalny", emotional, emotional_tomorrow)
evaluate("Biorytm intelektualny", intellectual, intellectual_tomorrow)

print("Sprawdź, jak się dziś czujesz i porównaj z wynikami! 😉")