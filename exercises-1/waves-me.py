import math
import datetime


def correct_input(predicates, message = "Please input a value: ", error = "Invalid value!"):
    while True:
        value = input(message)

        if check_predicates(predicates, value):
            return value

        else:
            print(error)


def check_predicates(predicates, value):
    for predicate in predicates:
        if not predicate(value):
            return False

    return True


def month_length(year, month):
    if ((month == 2) and ((year % 4 == 0) or ((year % 100 == 0) and (year % 400 == 0)))):
        return 29

    elif (month == 2):
        return 28

    elif (month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12):
        return 31

    else:
        return 30


def name_input(message, error):
    name = correct_input(
        [
            lambda value: isinstance(value, str),
            lambda value: len(value) > 0,
            lambda value: value.isalpha()
        ],
        message,
        error
    )

    return name


def dob_input(y_message, y_error, m_message, m_error, d_message, d_error):
    year = correct_input(
        [
            lambda value: isinstance(value, str),
            lambda value: value.isdigit(),
            lambda value: int(value) <= datetime.datetime.now().year,
            lambda value: int(value) >= 0,
        ],
        y_message,
        y_error
    )

    month = correct_input(
        [
            lambda value: isinstance(value, str),
            lambda value: value.isdigit(),
            lambda value: int(value) <= datetime.datetime.now().month,
            lambda value: int(value) > 0,
        ],
        m_message,
        m_error
    ) if year == datetime.datetime.now().year else correct_input(
        [
            lambda value: isinstance(value, str),
            lambda value: value.isdigit(),
            lambda value: int(value) <= 12,
            lambda value: int(value) > 0,
        ],
        m_message,
        m_error
    )

    day = correct_input(
        [
            lambda value: isinstance(value, str),
            lambda value: value.isdigit(),
            lambda value: int(value) <= datetime.datetime.now().day,
            lambda value: int(value) > 0,
        ],
        d_message,
        d_error
    ) if year == datetime.datetime.now().year and month == datetime.datetime.now().month else correct_input(
        [
            lambda value: isinstance(value, str),
            lambda value: value.isdigit(),
            lambda value: int(value) <= month_length(year, month),
            lambda value: int(value) > 0,
        ],
        d_message,
        d_error
    )

    date = datetime.date(int(year), int(month), int(day))

    return datetime.datetime(date.year, date.month, date.day)


def main():
    name = name_input(
        "Please enter your name: ",
        "Error: Invalid name!",
    )

    print(f"Hello, {name}!")

    dob = dob_input(
        "Please enter your year of birth: ",
        "Error: Invalid year!",
        "Please enter your month of birth: ",
        "Error: Invalid month!",
        "Please enter your day of birth: ",
        "Error: Invalid day!"
    )

    current_days = datetime.datetime.now() - dob

    print("You have lived for over: " + str(current_days.days))

    p_wave = math.sin((2 * math.pi * current_days.days) / 23)
    e_wave = math.sin((2 * math.pi * current_days.days) / 28)
    i_wave = math.sin((2 * math.pi * current_days.days) / 33)

    print(f"Your physical wave: {p_wave}")
    print(f"Your emotional wave: {e_wave}")
    print(f"Your intellectual wave: {i_wave}")

    if p_wave > 0.5:
        print("Seems like your physical wave is up! Make good use of your strength!")

    elif p_wave < -0.5:
        print("Seems like your physical wave is down! Try to save your strength!")

    if e_wave > 0.5:
        print("Looks like your emotional wave is up! Nothing will break your spirit!")

    elif e_wave < -0.5:
        print("Looks like your emotional wave is down! Keep being positive!")

    if i_wave > 0.5:
        print("Looks like your intellectual wave is up! Ideal to learn something new!")

    if e_wave < -0.5:
        print("Seems like your intellectual wave is down! Try to keep your mind clear!")

    if p_wave > 0.5 and e_wave > 0.5 and i_wave > 0.5:
        print("All your waves are up! Looks like this will be a wonderful day!")

    elif p_wave > 0.5 or e_wave > 0.5 or i_wave > 0.5:
        print("At least one of your waves is up! Make good use of it!")

    else:
        print("Oh no! Seems like all of your waves are down!")

        tomorrow_days = datetime.datetime.now() + datetime.timedelta(days = 1) - dob

        p_t_wave = math.sin((2 * math.pi * tomorrow_days.days) / 23)
        e_t_wave = math.sin((2 * math.pi * tomorrow_days.days) / 28)
        i_t_wave = math.sin((2 * math.pi * tomorrow_days.days) / 33)

        if p_t_wave > p_wave and e_t_wave > e_wave and i_t_wave > i_wave:
            print("Don't worry! All your waves will be better tomorrow!")

        elif p_t_wave > p_wave or e_t_wave > e_wave or i_t_wave > i_wave:
            print("Don't worry! Some of your waves will be better tomorrow!")

        else:
            print("Try to save your strength for tomorrow!")


main()

# Około dwie godziny.

