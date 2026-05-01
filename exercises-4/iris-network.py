import pandas
import numpy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import validation_curve

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt


def load_data():
    dataframe = pandas.read_csv("iris_big.csv")

    values = dataframe.iloc[:, :-1].values.astype(numpy.float32)
    species = dataframe.iloc[:, -1].values

    species = LabelEncoder().fit_transform(species)
    values = StandardScaler().fit_transform(values)

    values = torch.tensor(values, dtype=torch.float32)
    species = torch.tensor(species, dtype=torch.long)

    return TensorDataset(values, species)


class Net(nn.Module):

    # Architecture:
    # Network is made up of input layer, hidden and output.
    # - fc1 contains the input values
    # - ReLU is the hidden layer getting rid of linearity i and allowing to find more complex dependencies
    # - fc2 returns logits

    def __init__(self, input_size, hidden, classes):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def train(model, train_loader, validation_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 50
    train_losses = []
    validation_losses = []
    train_accuracy = []
    validation_accuracy = []

    # Training loop:
    # Each epoch is made up from two key parts:
    # 1) model.train(): Creating the gradients and learning
    # 2) model.eval(): Evaluation without modification

    # In the training part we:
    # - generate the gradients (loss.backward())
    # - update the weights (optimizer.step())

    # In the evaluation part:
    # - we do not create gradients (torch.no_grad())
    # - we evaluate the model predictions, based on a data it has no access to

    for _ in range(epochs):
        model.train()
        loss_sum = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * inputs.size(0)
            _, prediction = torch.max(out, 1)
            correct += (prediction == labels).sum().item()
            total += labels.size(0)

        train_losses.append(loss_sum / total)
        train_accuracy.append(correct / total)

        model.eval()
        loss_sum = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                out = model(inputs)
                loss = criterion(out, labels)

                loss_sum += loss.item() * inputs.size(0)
                _, prediction = torch.max(out, 1)
                correct += (prediction == labels).sum().item()
                total += labels.size(0)

        validation_losses.append(loss_sum / total)
        validation_accuracy.append(correct / total)

    return train_losses, validation_losses, train_accuracy, validation_accuracy


def plots(train_losses, validation_losses, train_accuracy, validation_accuracy):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.legend()

    plt.savefig("iris-network.png")
    plt.show()


def stats(model, validation_loader):
    model.eval()
    predictions = []
    labs = []

    with torch.no_grad():
        for inputs, labels in validation_loader:
            out = model(inputs)
            _, prediction = torch.max(out, 1)

            predictions.extend(prediction.numpy())
            labs.extend(labels.numpy())

    accuracy = accuracy_score(labs, predictions)
    matrix = confusion_matrix(labs, predictions)

    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", matrix)


    # Interpretation:
    # Accuracy informs, about the percent of validation samples, that have been correctly classified.
    # The confusion matrix informs which classes are confused with each other by the network.
    # High accuracy leads to the belief that the model comes to find the classes correctly in about 90% of cases.


def main():
    dataset = load_data()

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size

    train_set, validation_set = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=16)

    model = Net(dataset[:][0].shape[1], 10, len(numpy.unique(dataset[:][1])))

    train_losses, validation_losses, train_accuracy, validation_accuracy = train(model, train_loader, validation_loader)

    stats(model, validation_loader)
    plots(train_losses, validation_losses, train_accuracy, validation_accuracy)


main()