import pandas
import numpy
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt


def load_data():
    dataframe = pandas.read_csv("diagnosis.csv")

    values = dataframe.iloc[:, :-1].values.astype(numpy.float32)
    label = dataframe.iloc[:, -1].values

    label = LabelEncoder().fit_transform(label)
    values = StandardScaler().fit_transform(values)

    values = torch.tensor(values, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long)

    return TensorDataset(values, label)


class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        return self.net(x)


def train(model, train_loader, validation_loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    train_losses = []
    validation_losses = []
    train_accuracy = []
    validation_accuracy = []

    for _ in range(epochs):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0

        for inputs, label in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * inputs.size(0)
            _, prediction = torch.max(outputs, 1)
            correct += (prediction == label).sum().item()
            total += label.size(0)

        train_losses.append(loss_sum / total)
        train_accuracy.append(correct / total)

        model.eval()
        total = 0
        correct = 0
        loss_sum = 0

        with torch.no_grad():
            for inputs, label in validation_loader:
                outputs = model(inputs)
                loss = criterion(outputs, label)

                loss_sum += loss.item() * inputs.size(0)
                _, prediction = torch.max(outputs, 1)
                correct += (prediction == label).sum().item()
                total += label.size(0)

        validation_losses.append(loss_sum / total)
        validation_accuracy.append(correct / total)

    return train_losses, validation_losses, train_accuracy, validation_accuracy


def plots(train_losses, validation_losses, train_accuracy, validation_accuracy):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy, label="Train Accuracy")
    plt.plot(validation_accuracy, label="Validation Accuracy")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.savefig("diagnosis-network.png")
    plt.show()


def stats(model, validation_loader):
    model.eval()

    predictions = []
    labs = []

    with torch.no_grad():
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            _, prediction = torch.max(outputs, 1)
            predictions.extend(prediction.numpy())
            labs.extend(labels.numpy())

    accuracy = accuracy_score(labs, predictions)
    precision = precision_score(labs, predictions)
    recall = recall_score(labs, predictions)
    matrix = confusion_matrix(labs, predictions)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Confusion Matrix:\n", matrix)


def main():
    dataset = load_data()

    train_size = int(0.8 * len(dataset))
    validation_size = len(dataset) - train_size
    train_set, validation_set = random_split(dataset, [train_size, validation_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=32)

    model = Net(dataset[:][0].shape[1])

    train_losses, validation_losses, train_accuracy, validation_accuracy = train(model, train_loader, validation_loader)

    stats(model, validation_loader)
    plots(train_losses, validation_losses, train_accuracy, validation_accuracy)


main()