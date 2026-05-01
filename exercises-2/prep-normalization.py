import pandas
import matplotlib.pyplot as plt


def print_data(column):
    print(column.min())
    print(column.max())
    print(column.mean())
    print(column.std())


def get_data(dataframe):
    sepal_length = dataframe["sepal length (cm)"]
    sepal_width = dataframe["sepal width (cm)"]

    print_data(sepal_length)
    print_data(sepal_width)

    text_data = dataframe["target_name"]
    return sepal_length, sepal_width, text_data


def get_zcore_data(dataframe):
    sepal_length = dataframe["sepal length (cm)"]
    sepal_width = dataframe["sepal width (cm)"]

    sepal_length_zcore = (sepal_length - sepal_length.mean()) / sepal_length.std()
    sepal_width_zcore = (sepal_width - sepal_width.mean()) / sepal_width.std()

    return sepal_length_zcore, sepal_width_zcore


def get_minmax_data(dataframe):
    sepal_length = dataframe["sepal length (cm)"]
    sepal_width = dataframe["sepal width (cm)"]

    sepal_length_minmax = (sepal_length - sepal_length.min()) / (sepal_length.max() - sepal_length.min())
    sepal_width_minmax = (sepal_width - sepal_width.min()) / (sepal_width.max() - sepal_width.min())

    return sepal_length_minmax, sepal_width_minmax


def plot(sepal_length, sepal_width, text_data, file):
    colors = {
        "setosa": "red",
        "versicolor": "green",
        "virginica": "blue"
    }

    for species in text_data.unique():
        idx = text_data == species
        plt.scatter(
            sepal_length[idx],
            sepal_width[idx],
            c = colors[species],
            label = species,
            s = 20,
            alpha = 0.7
        )

    plt.xlabel("Sepal length (cm)")
    plt.ylabel("Sepal width (cm)")
    plt.title("Original data")
    plt.legend(title="Species")

    plt.savefig(file)
    plt.show()

    plt.close()


def main():
    dataset = pandas.read_csv("iris_big.csv")
    sepal_length, sepal_width, text_data = get_data(dataset)
    sepal_length_zcore, sepal_width_zcore = get_zcore_data(dataset)
    sepal_length_minmax, sepal_width_minmax = get_minmax_data(dataset)
    plot(sepal_length, sepal_width, text_data, "prep-normalization-1.png")
    plot(sepal_length_zcore, sepal_width_zcore, text_data, "prep-normalization-2.png")
    plot(sepal_length_minmax, sepal_width_minmax, text_data, "prep-normalization-3.png")


main()