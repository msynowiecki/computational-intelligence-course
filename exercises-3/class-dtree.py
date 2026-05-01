import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix
from sklearn import tree as sktree
import matplotlib.pyplot as plt


def split_data(dataframe):
    train_set, test_set = train_test_split(dataframe.values, train_size=0.7, random_state=300865)

    print(train_set)
    print(test_set)

    return train_set, test_set


def train_tree(train_set):
    numeric_train = train_set[:, 0:4]
    species_train = train_set[:, 4]

    tree = DecisionTreeClassifier()
    tree.fit(numeric_train, species_train)

    return tree


def show_tree(tree, feature_names):
    rules = export_text(tree, feature_names=feature_names)
    print(rules)

    plt.figure(figsize=(40, 24))
    sktree.plot_tree(
        tree,
        feature_names=feature_names,
        class_names=tree.classes_,
        filled=True
    )
    plt.savefig("class-dtree.png")
    plt.show()


def test_tree(test_set, tree):
    numeric_test = test_set[:, 0:4]
    species_test = test_set[:, 4]

    predictions = tree.predict(numeric_test)

    good = 0
    length = test_set.shape[0]

    for index in range(length):
        if predictions[index] == species_test[index]:
            good = good + 1

    print("Correct:", good)
    print("Accuracy:", good / length * 100, "%")

    print("Accuracy (score):", tree.score(numeric_test, species_test))

    confusion = confusion_matrix(species_test, predictions)
    print("Confusion matrix:")
    print(confusion)


def main():
    dataframe = pd.read_csv("iris_big.csv")

    train_set, test_set = split_data(dataframe)

    tree = train_tree(train_set)

    feature_names = list(dataframe.columns[0:4])
    show_tree(tree, feature_names)

    test_tree(test_set, tree)


main()