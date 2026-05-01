import pandas

# Problems:
# Unnecessary quotation marks causing the csv to be read as one column
# Unnecessary quotation marks create an unnecessary fifth column
# Unnecessary quotation marks prevent reading the file due to wrong header format
# Mistakes in species names like unnecessary special symbols or wrong types
# Minor spelling mistakes: two options
# - Drop all mistakes
# - Print all unique, and try to fix them using a dictionary


def base_correction(dataframe):
    dataframe = (dataframe[0]
                 .str.replace('"', '', regex=False)
                 .str.split(',', expand=True))

    dataframe = (dataframe
                 .drop(columns=[5])
                 .drop(0))

    dataframe.columns = [
        'sepal length (cm)',
        'sepal width (cm)',
        'petal length (cm)',
        'petal width (cm)',
        'target_name']

    return dataframe


def number_correction(dataframe):
    for column in ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']:
        dataframe[column] = pandas.to_numeric(dataframe[column], errors='coerce')

    return dataframe


def species_correction(dataframe):
    dataframe['target_name'] = dataframe['target_name'].str.strip()
    dataframe['target_name'] = dataframe['target_name'].str.lower()
    dataframe['target_name'] = dataframe['target_name'].str.replace('iris', '', regex=False)
    dataframe['target_name'] = dataframe['target_name'].str.replace('[^a-z]', '', regex=True)

    correct_species = ['setosa', 'versicolor', 'virginica']
    dataframe['target_name'] = dataframe['target_name'].where(dataframe['target_name'].isin(correct_species))

    return dataframe

    # match


def value_correction(dataframe):

    for column in ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']:

        for species in dataframe['target_name'].unique():

            mask_species = dataframe['target_name'] == species
            mask_range = (dataframe[column] > 0) & (dataframe[column] < 15)

            median_value = dataframe.loc[mask_species & mask_range, column].median()

            dataframe.loc[mask_species, column] = dataframe.loc[mask_species, column].where(mask_range, median_value)

    return dataframe


def nan_removal(dataframe):
    return dataframe.dropna().reset_index(drop=True)


def show_info(dataframe):
    print(dataframe)
    print(dataframe.isnull().sum())
    print(dataframe.describe())
    print(dataframe.info())
    print(dataframe['target_name'].unique())


def main():
    dataset = pandas.read_csv("iris_big_with_errors.csv", header=None)
    dataset = base_correction(dataset)
    dataset = number_correction(dataset)
    dataset = species_correction(dataset)
    dataset = value_correction(dataset)
    dataset = nan_removal(dataset)
    show_info(dataset)


main()