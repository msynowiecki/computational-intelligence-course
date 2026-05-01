import pandas
from sklearn.model_selection import train_test_split


def classify_iris(sepal_length, sepal_width, petal_length, petal_width):
    if sepal_length > 4:
        return("setosa")
    elif petal_length <= 5:
        return("virginica")
    else:
        return("versicolor")


def classify_iris_better(sepal_length, sepal_width, petal_length, petal_width):
    if petal_length < 2.55:
        return("setosa")
    elif petal_width > 1.65:
        return("virginica")
    else:
        return("versicolor")


def test_classification(test_set, test_func):
    good = 0
    length = test_set.shape[0]

    for index in range(length):
        if test_func(test_set[index, 0], test_set[index, 1], test_set[index, 2], test_set[index, 3]) == test_set[index, 4]:
            good = good + 1

    print(good)
    print(good / length * 100, "%")


def main():
    dataframe = pandas.read_csv("iris_big.csv")
    train_set, test_set = train_test_split(dataframe.values, train_size=0.7, random_state=300865)
    test_classification(test_set, classify_iris)
    test_classification(test_set, classify_iris_better)



main()