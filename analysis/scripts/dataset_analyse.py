from collections import Counter
from optparse import Option
from typing import List, Optional, Tuple
from IPython.display import display
from matplotlib.pyplot import plot, show, subplots, xlabel, xticks, ylabel, suptitle
import matplotlib.cm as cm
from numpy import percentile, round, arange
from pandas import DataFrame, Series
from scipy.sparse import data
from seaborn import histplot, boxenplot
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture



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
    

def biplot(reduced_dataframe: DataFrame, pca: Optional[PCA] = None, original_dataframe: Optional[DataFrame] = None, cluster_column: Optional[str] = None) -> None:
    """
    Biplot is the scatter chart of the PCA. It has 3 pricipal modes : 
    - Basic scatter mode
    - Arrow with original features
    - different colors by clusters
    Input:
        reduced_dataframe, Dataframe = the dataframe reduced in 2 principal components
        pca, Optional[PCA] = None : The PCA used to reduce dataframe, to get original features
        original_dataframe, Optional[Dataframe] = None : The original dataframe, to get original features
        cluster_columns, Optional[str] = None : The name of the cluster column, to get different colors by clusters
    """
    _, ax = subplots(figsize = (14,8))  

    if cluster_column:
        ax.scatter(x=reduced_dataframe.loc[:, 'principal_component_1'], y=reduced_dataframe.loc[:, 'principal_component_2'], 
            facecolors='b', edgecolors='b', s=70, alpha=0.5, c=reduced_dataframe[cluster_column])
    else:
        ax.scatter(x=reduced_dataframe.loc[:, 'principal_component_1'], y=reduced_dataframe.loc[:, 'principal_component_2'], 
            facecolors='b', edgecolors='b', s=70, alpha=0.5)
    
    if original_dataframe is not None and pca is not None:
        feature_vectors = pca.components_.T
        arrow_size, text_pos = 7.0, 8.0,

        for feature_index, vector in enumerate(feature_vectors):
            ax.arrow(0, 0, arrow_size*vector[0], arrow_size*vector[1], 
                    head_width=0.2, head_length=0.2, linewidth=2, color='red')
            ax.text(vector[0]*text_pos, vector[1]*text_pos, original_dataframe.columns[feature_index], color='black', 
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


def number_clusters(dataframe: DataFrame, clusters: int = 7) -> None:
    scores = {}
    for cluster in range(2, clusters):
        clusterer = GaussianMixture(random_state=6, n_components=cluster)
        clusterer.fit(dataframe)

        predictions = clusterer.predict(dataframe)

        score = silhouette_score(dataframe, predictions)
        scores[cluster] = score
        print(cluster, ' : Silhouette score is: ' + str(score), '\n')
    
    print('All Scores : ' + str(scores))


def boxplot(dataframe: DataFrame, cluster_column: str):
    for column in dataframe.columns:
        boxenplot(x=dataframe[cluster_column], y=dataframe[column])
        show()


def silhouette_plot(dataframe: DataFrame, n_clusters: int) -> None:
    fig, (ax1, ax2) = subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(dataframe) + (n_clusters + 1) * 10])

    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(dataframe)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(dataframe, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    sample_silhouette_values = silhouette_samples(dataframe, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(
            arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            alpha=0.7,
        )

        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
        )

    show()