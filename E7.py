import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

#Importing the Dataset
df = pd.read_csv('/content/car_data.csv')
df

from sklearn.cluster import KMeans
relevant_cols = ['Kilometers Driven', 'Selling Price', 'Year', 'Owner']
df = df[relevant_cols]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

def find_best_clusters(df, maximum_K):
  cluster_centers = []
  k_values = []

  for k in range(1, maximum_K):
    kmeans_model = KMeans(n_clusters = k)
    kmeans_model.fit(df)
    cluster_centers.append(kmeans_model.inertia_)
    k_values.append(k)

  return cluster_centers, k_values

def generate_elbow_plot(cluster_centers, k_values):
  figure = plt.subplots(figsize = (12,6))
  plt.plot(k_values, cluster_centers, 'o-', color = 'blue')
  plt.xlabel('Number of Clusters(K)')
  plt.ylabel('Cluster Inertia')
  plt.title('Elbow Plot of KMeans')
  plt.show()

cluster_centers, k_values = find_best_clusters(scaled_data, 12)
generate_elbow_plot(cluster_centers, k_values)

kmeans_model = KMeans(n_clusters = 5)
kmeans_model.fit(scaled_data)

df['clusters'] = kmeans_model.labels_
df.head()

kilo_driven = df['Kilometers Driven']
sell_price = df['Selling Price']
clusters = df['clusters']
plt.scatter(kilo_driven[: 200], sell_price[:200], c = clusters[:200])



#DBSCAN start
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#Choose features for clustering
X = df[['Selling Price', 'Kilometers Driven']]

#Scale Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Apply DBSCAN
dbscan = DBSCAN(eps = 1.5, min_samples = 1000)
clusters = dbscan.fit_predict(X_scaled)

#Plot Outliers 
outliers = X[clusters == -1]
plt.scatter(outliers['Kilometers Driven'], outliers['Selling Price'], c = 'r')
plt.title('DBSCAN Outliers')
plt.xlabel('Kilometers Driven')
plt.ylabel('Selling Price')
plt.show()







#Alternate Code for K-Means

from sklearn.cluster import KMeans
kmeans = df[['Kilometers Driven', 'Selling Price']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(kmeans)

#Statisitcs of Scaled Data
pd.DataFrame(data_scaled).describe()

kmeans_algo = KMeans(n_clusters = 2, init = 'k-means++')

#Fitting the K-means Algorithm on Scaled Data
kmeans_algo.fit(data_scaled)

kmeans_algo.inertia_

#Fitting multiple K-means algorithms and stroing the values in an empty list
SSE = []
for cluster in range(1,20):
  kmeans_algo = KMeans(n_clusters = cluster, init = 'k-means++')
  kmeans_algo.fit(data_scaled)
  SSE.append(kmeans_algo.inertia_)

  #Converting the results into a DataFrame and plotting them
  frame = pd.DataFrame({'Cluster': range(1,20), 'SSE': SSE })
  plt.figure(figsize=(12,6))
  plt.plot(frame['Cluster'], frame['SSE'], marker = 'o')
  plt.xlabel('Number of Clusters')
  plt.ylabel('Inertia')

#K Means using 5 clusters and k-means++ initialization
kmeans_algo = KMeans(n_clusters = 5, init = 'k-means++')
kmeans_algo.fit(data_scaled)
pred = kmeans_algo.predict(data_scaled)

frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()