import math
import datetime


def correct_input(predicates, message="Please input a value: ", error="Invalid value!"):
    while True:
        value = input(message)

        if all(predicate(value) for predicate in predicates):
            return value
        else:
            print(error)


def name_input(message, error):
    return correct_input(
        [
            lambda value: len(value) > 0,
            lambda value: value.isalpha()
        ],
        message,
        error
    )


def dob_input():
    current_date = datetime.date.today()

    while True:
        try:
            year = int(input("Please enter your year of birth: "))
            month = int(input("Please enter your month of birth: "))
            day = int(input("Please enter your day of birth: "))

            birth_date = datetime.date(year, month, day)

            if birth_date > current_date:
                print("Error: Birth date cannot be in the future!")
                continue

            return birth_date

        except ValueError:
            print("Error: Invalid date! Please try again.")


def calculate_wave(days, cycle):
    return math.sin((2 * math.pi * days) / cycle)


def main():
    name = name_input(
        "Please enter your name: ",
        "Error: Invalid name!"
    )

    print(f"\nHello, {name}!")

    dob = dob_input()

    today = datetime.date.today()
    days_lived = (today - dob).days

    print(f"\nYou have lived for over: {days_lived} days")

    p_wave = calculate_wave(days_lived, 23)
    e_wave = calculate_wave(days_lived, 28)
    i_wave = calculate_wave(days_lived, 33)

    print(f"\nYour physical wave: {p_wave:.2f}")
    print(f"Your emotional wave: {e_wave:.2f}")
    print(f"Your intellectual wave: {i_wave:.2f}")

    # Physical
    if p_wave > 0.5:
        print("Physical wave is high! Use your strength!")
    elif p_wave < -0.5:
        print("Physical wave is low! Save your energy!")

    # Emotional
    if e_wave > 0.5:
        print("Emotional wave is high! Great mood ahead!")
    elif e_wave < -0.5:
        print("Emotional wave is low! Stay positive!")

    # Intellectual
    if i_wave > 0.5:
        print("Intellectual wave is high! Perfect time to learn!")
    elif i_wave < -0.5:
        print("Intellectual wave is low! Try to focus calmly!")

    # General summary
    if p_wave > 0.5 and e_wave > 0.5 and i_wave > 0.5:
        print("\nAll waves are high! Wonderful day ahead!")
    elif p_wave > 0.5 or e_wave > 0.5 or i_wave > 0.5:
        print("\nAt least one wave is high! Use it wisely!")
    else:
        print("\nAll waves are low today...")

        tomorrow = today + datetime.timedelta(days=1)
        tomorrow_days = (tomorrow - dob).days

        p_t = calculate_wave(tomorrow_days, 23)
        e_t = calculate_wave(tomorrow_days, 28)
        i_t = calculate_wave(tomorrow_days, 33)

        if p_t > p_wave and e_t > e_wave and i_t > i_wave:
            print("But tomorrow will be better in every way!")
        elif p_t > p_wave or e_t > e_wave or i_t > i_wave:
            print("Some waves improve tomorrow!")
        else:
            print("Rest well and try again tomorrow!")


if __name__ == "__main__":
    main()