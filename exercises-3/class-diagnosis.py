import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

def plot_3d(dataframe):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    healthy = dataframe[dataframe['diagnosis'] == 0]
    sick = dataframe[dataframe['diagnosis'] == 1]

    ax.scatter(healthy['param1'], healthy['param2'], healthy['param3'], label="healthy", c='blue')
    ax.scatter(sick['param1'], sick['param2'], sick['param3'], label="sick", c='red')

    ax.set_xlabel("param1")
    ax.set_ylabel("param2")
    ax.set_zlabel("param3")

    plt.legend()
    plt.savefig("class-diagnosis.png")
    plt.show()

def train_model(train_set, model):
    numeric_train = train_set[:, 0:3]
    label_train = train_set[:, 3]
    model.fit(numeric_train, label_train)
    return model

def test_model(test_set, model):
    numeric_test = test_set[:, 0:3]
    label_test = test_set[:, 3]

    predictions = model.predict(numeric_test)

    correct = sum(predictions[index] == label_test[index] for index in range(len(label_test)))
    accuracy = correct / len(label_test)

    precision = precision_score(label_test, predictions)
    recall = recall_score(label_test, predictions)

    confusion = confusion_matrix(label_test, predictions)

    print("Accuracy:", round(accuracy, 3))
    print("Precision:", round(precision, 3))
    print("Recall:", round(recall, 3))
    print("Confusion matrix:")
    print(confusion)

    return accuracy, precision, recall

def main():
    dataframe = pd.read_csv("diagnosis.csv")

    plot_3d(dataframe)

    train_set, test_set = train_test_split(dataframe.values, train_size=0.7, random_state=300865)

    results = {}

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "kNN k=3": KNeighborsClassifier(n_neighbors=3),
        "kNN k=5": KNeighborsClassifier(n_neighbors=5),
        "kNN k=11": KNeighborsClassifier(n_neighbors=11),
        "Naive Bayes": GaussianNB(),
        "MLP": MLPClassifier(max_iter=1000)
    }

    for name in models:
        model = train_model(train_set, models[name])
        results[name] = test_model(test_set, model)

    for name in results:
        print(name + ":", results[name])

main()

# Accuracy - Percentage of correctly labeled instances.
# Precision - How many people labeled as sick are really sick
# Recall / Sensitivity - How many sick have been correctly identified

# In case there are a lot of elements from single pool, the classificator that always guesses the majority might have a large accuracy.
# In those scenarios precision and recall tell us more about the predictions.