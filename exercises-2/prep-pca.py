import pandas
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_data(dataframe):
    numeric_data = dataframe[["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]]
    text_data = dataframe["target_name"]
    return numeric_data, text_data


def base_pca(numeric_data):
    pca = PCA()
    base = pca.fit(numeric_data)
    return base


def reduced_pca(numeric_data):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(numeric_data)
    return reduced


def plot(pca, text_data):
    colors = {
        "setosa": "red",
        "versicolor": "green",
        "virginica": "blue"
    }

    for species in text_data.unique():
        idx = text_data == species
        plt.scatter(
            pca[idx, 0],
            pca[idx, 1],
            c = colors[species],
            label = species,
            s = 20,
            alpha = 0.7
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA - Iris dataset")
    plt.legend(title="Species")

    plt.savefig("prep-pca.png")
    plt.show()


def main():
    dataset = pandas.read_csv("iris_big.csv")
    numeric_data, text_data = get_data(dataset)
    base = base_pca(numeric_data)
    reduced = reduced_pca(numeric_data)
    plot(reduced, text_data)


main()

# Explanation:
# PC1 is 0.92623457
# PC2 is 0.04895082
# PC3 is 0.01789868
# PC4 is 0.00691592
# PC1 + PC2 = 0.92623457 + 0.04895082 = 0.97518539 > 0.95
# That means that PC1 and PC2 explain more than 95% of the data.
# That means that by ignoring PC3 and PC4 the data loss is less than 5%.