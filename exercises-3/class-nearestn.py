import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix


def split_data(dataframe):
    train_set, test_set = train_test_split(dataframe.values, train_size=0.7, random_state=300865)

    print(train_set)
    print(test_set)

    return train_set, test_set


def train_model(train_set, model):
    numeric_train = train_set[:, 0:4]
    species_train = train_set[:, 4]

    model.fit(numeric_train, species_train)

    return model


def test_model(test_set, model):
    numeric_test = test_set[:, 0:4]
    species_test = test_set[:, 4]

    predictions = model.predict(numeric_test)

    good = 0
    length = test_set.shape[0]

    for index in range(length):
        if predictions[index] == species_test[index]:
            good = good + 1

    accuracy = good / length * 100

    print("Correct:", good)
    print("Accuracy:", accuracy, "%")

    print("Accuracy (score):", model.score(numeric_test, species_test))

    matrix = confusion_matrix(species_test, predictions)
    print("Confusion matrix:")
    print(matrix)

    return accuracy


def main():
    dataframe = pd.read_csv("iris_big.csv")

    train_set, test_set = split_data(dataframe)

    results = {}

    tree = DecisionTreeClassifier()
    tree = train_model(train_set, tree)
    results["Decision Tree"] = test_model(test_set, tree)

    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn3 = train_model(train_set, knn3)
    results["kNN k=3"] = test_model(test_set, knn3)

    knn5 = KNeighborsClassifier(n_neighbors=5)
    knn5 = train_model(train_set, knn5)
    results["kNN k=5"] = test_model(test_set, knn5)

    knn11 = KNeighborsClassifier(n_neighbors=11)
    knn11 = train_model(train_set, knn11)
    results["kNN k=11"] = test_model(test_set, knn11)

    nb = GaussianNB()
    nb = train_model(train_set, nb)
    results["Naive Bayes"] = test_model(test_set, nb)

    mlp = MLPClassifier(max_iter=1000)
    mlp = train_model(train_set, mlp)
    results["MLP"] = test_model(test_set, mlp)

    for key in results:
        print(key + ":", results[key], "%")

    best = max(results, key=results.get)
    print("Best:", best)


main()