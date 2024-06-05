from collections import Counter
from typing import List, Optional, Tuple
from IPython.display import display
from matplotlib.pyplot import show
from numpy import percentile
from pandas import DataFrame
from seaborn import histplot


def print_unique_values(dataframe: DataFrame, columns: List[str]) -> None:
    for column in columns:
        print(
            f"the unique values for the column {column} are : \n{dataframe[column].unique()}\n"
        )


def counter_columns(dataframe: DataFrame, columns: List[str]) -> None:
    """ """
    for column in columns:
        print(f"The counter of {column} is : {Counter(dataframe[column])}")


def lowest_and_biggest_correlation(matrix: DataFrame) -> Tuple[int, int]:
    unstack_matrix = matrix.unstack()
    print(f"The lowest correlation is {unstack_matrix.sort_values().iloc[0]}")
    index = 0
    while index < len(unstack_matrix):
        if unstack_matrix.sort_values(ascending=False).iloc[index] != 1.0:
            print(
                f"The biggest correlation is {unstack_matrix.sort_values(ascending=False).iloc[index]}"
            )
            break
        index += 1
    return (
        unstack_matrix.sort_values().iloc[0],
        unstack_matrix.sort_values(ascending=False).iloc[index],
    )


def hisplot_columns(dataframe: DataFrame, columns: List[str]) -> None:
    for column in dataframe.columns:
        if dataframe[column].dtype in ("int64", "float64"):
            histplot(data=dataframe, x=column)
            show()


def find_dataframe_outliers(dataframe: DataFrame) -> List[List[Optional[int]]]:
    outliers = []

    for column in dataframe.columns:
        if dataframe[column].dtype not in ['int64', 'float64']:
            continue
        first_quartile = percentile(dataframe[column], 5)
        third_quartile = percentile(dataframe[column], 95)
        step = 1.5 * (third_quartile - first_quartile)

        outliers_per_column = dataframe[
            ~(
                (dataframe[column] >= first_quartile - step)
                & (dataframe[column] <= third_quartile + step)
            )
        ]
        print(f"Data points considered outliers for the column {column}:")
        display(outliers_per_column)
        lista = outliers_per_column.index.tolist()
        outliers.append(lista)
    return outliers