from collections import Counter
from typing import List, Optional, Tuple
from IPython.display import display
from matplotlib.pyplot import plot, show, subplots, xlabel, xticks, ylabel
from numpy import percentile, round
from pandas import DataFrame, Series
from seaborn import histplot
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def print_unique_values(dataframe: DataFrame, columns: List[str]) -> None:
    for column in columns:
        print(
            f"the unique values for the column {column} are : \n{dataframe[column].unique()}\n"
        )


def counter_columns(dataframe: DataFrame, columns: List[str]) -> None:
    for column in columns:
        print(f"The counter of {column} is : {Counter(dataframe[column])}")


def lowest_and_biggest_correlation(matrix: DataFrame) -> Tuple[int, int]:
    unstack_matrix = matrix.unstack()
    if not isinstance(unstack_matrix, Series):
        raise ValueError

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
    for column in columns:
        if dataframe[column].dtype in ("int64", "float64"):
            histplot(data=dataframe, x=column)
            show()


def find_dataframe_outliers(
    dataframe: DataFrame, columns: List[str], percent: int = 75
) -> List[List[Optional[int]]]:
    """
    Tukey's method to identify outliers, with custom percentile (default is quartile):
    The method detects outliers as 1.5 * the difference between the first and the third quartile.

    Input :
        dataframe, DataFrame: The dataset
        columns, List[str] : The list of columns to detect outliers
        percent, int = 75 : The third quartile percent (1-percent first quartile)
    Output :
        List[List[Optional[int]]] : The matrix of list of outliers' index
    Mathematic expression :
        - s = (3*(q3-q1)) / 2
        - q3 + s > outliers or outliers < q1 - s
    """
    outliers = []

    for column in columns:
        if dataframe[column].dtype not in ["int64", "float64"]:
            continue
        first_quartile = percentile(dataframe[column], 100 - percent)
        third_quartile = percentile(dataframe[column], percent)
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


def pca_visualize_categories(dataframe: DataFrame, pca: PCA) -> None:

    n_dimensions = [f'Dimension {dimension}' for dimension in range(1, len(pca.components_)+1)]
    components = DataFrame(round(pca.components_, 4), columns = list(dataframe.keys()))

    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = DataFrame(round(ratios, 4), columns = ['Explained Variance'])

    fig, ax = subplots(figsize = (14,8))
    components.plot(ax=ax, kind='bar')
    ax.set_ylabel("Feature weights")
    ax.set_xticklabels( n_dimensions, rotation=0)

    for index, variance in enumerate(pca.explained_variance_ratio_):
	    ax.text(index-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(variance)) 
    

def biplot(good_data: DataFrame, reduced_data: DataFrame, pca: PCA) -> None:

    _, ax = subplots(figsize = (14,8))   
    ax.scatter(x=reduced_data.loc[:, 'principal_component_1'], y=reduced_data.loc[:, 'principal_component_2'], 
        facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    feature_vectors = pca.components_.T

    # we use scaling factors to make the arrows easier to see
    arrow_size, text_pos = 7.0, 8.0,

    # projections of the original features
    for feature_index, vector in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*vector[0], arrow_size*vector[1], 
                  head_width=0.2, head_length=0.2, linewidth=2, color='red')
        ax.text(vector[0]*text_pos, vector[1]*text_pos, good_data.columns[feature_index], color='black', 
                 ha='center', va='center', fontsize=18)

    ax.set_xlabel("principal_component_1", fontsize=14)
    ax.set_ylabel("principal_component_2", fontsize=14)
    ax.set_title("PC plane with original feature projections.", fontsize=16);
    return ax


def elbow_chart(dataframe: DataFrame) -> None:
    sum_squared_error = []
    for cluster in range(1, 10):
        kmeans = KMeans(n_clusters=cluster,)
        kmeans.fit(dataframe)
        sum_squared_error.append(kmeans.inertia_)

    plot(range(1, 10), sum_squared_error)
    xticks(range(1, 11))
    xlabel("Number of Clusters")
    ylabel("SSE")
    show()