# Customer personality analysis

## Unsupervised classification algorithm
Clustering, which is an unsupervised classification method, brings together a set of learning algorithms whose goal is to group together unlabeled data with similar properties.
We will see three clustering methods : 
- k-Means clustering
- DBSCAN (Density-based spatial clustering of applications with noise)
- AHC (Hierarchical Clustering Algorithm)

  
### k-Means clustering
K-means is a centroid-based clustering algorithm, where we calculate the distance between each data point and a centroid to assign it to a cluster. The goal is to identify the K number of groups in the dataset. 

It is an iterative process of assigning each data point to the groups and slowly data points get clustered based on similar features. The objective is to minimize the sum of distances between the data points and the cluster centroid, to identify the correct group each data point should belong to. 

Here, we divide a data space into K clusters and assign a mean value to each. The data points are placed in the clusters closest to the mean value of that cluster. There are several distance metrics available that can be used to calculate the distance. 

![k-means representation](./media/1_rw8IUza1dbffBhiA4i0GNQ.png)


### DBSCAN (Density-based spatial clustering of applications with noise)
DBSCAN is a base algorithm for density-based clustering. It can discover clusters of different shapes and sizes from a large amount of data, which is containing noise and outliers.

K-Means clustering may cluster loosely related observations together. Every observation becomes a part of some cluster eventually, even if the observations are scattered far away in the vector space. Since clusters depend on the mean value of cluster elements, each data point plays a role in forming the clusters. A slight change in data points might affect the clustering outcome. This problem is greatly reduced in DBSCAN due to the way clusters are formed. This is usually not a big problem unless we come across some odd shape data.

Another challenge with k-means is that you need to specify the number of clusters (“k”) in order to use it. Much of the time, we won’t know what a reasonable k value is a priori.

What’s nice about DBSCAN is that you don’t have to specify the number of clusters to use it. All you need is a function to calculate the distance between values and some guidance for what amount of distance is considered “close”. DBSCAN also produces more reasonable results than k-means across a variety of different distributions. Below figure illustrates the fact:

![dbscan vs k-means](./media/0_xu3GYMsWu9QiKNOo.png)

### AHC (Hierarchical Clustering Algorithm)

The principle of AHC is to group individuals together according to a pre-defined similarity criterion, expressed in the form of a distance matrix, expressing the distance between each individual taken in pairs. Two identical observations will have a distance of zero. The more dissimilar the two observations, the greater the distance. CAH then iteratively combines the individuals to produce a dendrogram or classification tree. Classification is bottom-up, as it starts from individual observations; it is hierarchical, as it produces progressively larger classes or groups, including sub-groups within them. By cutting this tree at a certain height, we produce the desired partition.

## Determining the optimal number of clusters
We do not know how many clusters there are in our data. Indeed, figuring out how many clusters there are may be the reason why we want to perform clustering in the first place. Certainly, domain knowledge of the data set may help determine the number of clusters. However, this assumes that you know the target classes (or at least how many classes there are), and this is not true in unsupervised learning. We need a method that informs us about the number of clusters without relying on a target variable.

One possible solution in determining the correct number of clusters is a brute-force approach. We try applying a clustering algorithm with different numbers of clusters. Then, we find the magic number that optimizes the quality of the clustering results. In this article, we first introduce two popular metrics to assess cluster quality. We then cover three approaches to find the optimal number of clusters:
- The elbow method
- The optimization of the silhouette coefficient
- The gap statistic

## Measuring clustering quality
We have a choice of several methods for measuring clustering quality. In general, these methods can be divided into two groups, depending on whether the ground truth is available or not. Here, ground truth is the ideal clustering, often constructed with the help of human experts.

If the ground truth is available, it can be used by extrinsic methods, which compare the grouping with the truth and the group measure. If ground truth is not available, we can use intrinsic methods , which assess the quality of a clustering by considering how far the clusters are separated. Ground truth can be seen as supervision in the form of “cluster labels”. Consequently, extrinsic methods are also called supervised methods, while intrinsic methods are unsupervised methods.

sources : 
- https://dataanalyticspost.com/Lexique/clustering/#:~:text=Clustering%2(ou%20partitionnement%20des%20donn%C3%A9es,%C3%A9tiquet%C3%A9es%20pr%C3%A9sentant%20des%20propri%C3%A9t%C3%A9s%20similaires.
- https://neptune.ai/blog/k-means-clustering#:~:text=%E2%80%9CK%2Dmeans%20clustering%20is%20a,Source
- https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html
- https://freedium.cfd/https://towardsdatascience.com/how-many-clusters-6b3f220f0ef5
- https://larmarange.github.io/guide-R/analyses_avancees/classification-ascendante-hierarchique.html
- https://www.sciencedirect.com/topics/computer-science/clustering-quality