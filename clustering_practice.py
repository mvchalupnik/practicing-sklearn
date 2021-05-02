# Import module
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import numpy as np
import matplotlib.pyplot as plt
# Import hierarchical clustering libraries
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


cancer = load_breast_cancer()
X = cancer.data
y = cancer.target
print(cancer.keys())
print(cancer.feature_names)
print("There are {} features".format(len(cancer.feature_names)))
print("There are {} observations".format(len(cancer.data)))

#Generally clustering makes more sense to do when your number of observations is small 
#compared to your number of features

X_scaled = scale(X)


############### Practice KMeans clustering method
#Choose initial centroids randomly, assign each observation to the nearest centroid, create new centroids by taking
#the mean values of all observations assigned to a given centroid, iterate until centroids do not significantly move

# Instantiate
kmeans = KMeans(n_clusters=2, random_state=123)
# Fit
fit = kmeans.fit(X_scaled)
# Print inertia: sum of squared distances to closest cluster center
print("Sum of squared distances for 2 clusters is", kmeans.inertia_)
kmeans = KMeans(n_clusters=10, random_state=123)
# Fit
fit = kmeans.fit(X_scaled)
# Print inertia: sum of squared distances to closest cluster center
print("Sum of squared distances for 10 clusters is", kmeans.inertia_)

############## Hierarchical agglomerative clustering
#Make dendrograms -- tree diagrams that connect each datapoint by distance. 
#Draw perpendicular lines through the dendrogram to select out the groups

# Create dendrogram
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
#What is y axis? Measure of closeness of either individual data points or clusters
plt.show()
# Create clusters and fit
hc = AgglomerativeClustering(affinity = 'euclidean', linkage = 'ward')
hc.fit(X_scaled)

# Print number of clusters
print(hc.n_clusters_)


############## Determining K 
#In general, two methods for determining K: silhouette method and elbow method
#Silhouette method uses silhouette coefficient, composed of mean distance between observation and all others in 
#same cluster, and mean distance between each observation and all others in next nearest cluster. 
#1 is good, means observation is close to others in same cluster. -1 is bad. 

#Elbow method - plot the sum of the square distance from each observation to the centroid against the number
#of clusters. The "elbow point" on the plot will be the optimal k point. 

# Silhouette method
for n_clusters in range(2,9):
    kmeans = KMeans(n_clusters=n_clusters)
    # Fit and predict your k-Means object
    preds = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {})".format(n_clusters, score))



#Elbow method
# Create empty list
sum_of_squared_distances = []

# Create for loop
for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(X)
    sum_of_squared_distances.append(kmeans.inertia_)

# Plot
plt.plot(range(1,15), sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()

#Ideally, Elbow method and Silhouette method should "agree on" the ideal k!
