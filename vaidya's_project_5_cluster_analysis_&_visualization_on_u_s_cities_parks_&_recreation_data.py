# -*- coding: utf-8 -*-
"""Vaidya's Project 5: Cluster Analysis & Visualization on U.S. Cities Parks & Recreation Data.ipynb

I would like to acknowledge the use of Stack Overflow and the numerous resources available via Google insights and code examples as well as the Google Colab Python Helpdesk, which helped me resolve technical challenges that came up during this project. Additionally, I did refer to some of the modules in datacamp (the ones on box plot) and combining columns.

##Part 1: Cluster analysis of parks & facilities data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans

# Loading the dataset
from google.colab import files
uploaded = files.upload("file path")

#Step 1

df = pd.read_csv("file path")

# Displaying the first few rows
print(df.head())

# Checking dataset dimensions, info, and summary statistics for the first step
print(df.shape)
df.info()
print(df.describe())

# Step 2

# Selecting numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols]

# Normalizing the numeric features (each value divided by the column standard deviation)
df_normalized = df_numeric / df_numeric.std()

# Displaying the first few rows of the normalized data
print(df_normalized.head())

print("Available columns:", df.columns.tolist())

# Step 3
# Part 1

#The dendogram
# Choosing two features for hierarchical clustering
features_hierarchy = ["Parks per 10,000 residents", "Acres per 1,000 people"]
data_hierarchy = df_normalized[features_hierarchy]

# Computing the linkage matrix using the 'ward' method
Z = linkage(data_hierarchy, method="ward")

# Plotting the dendrogram
plt.figure(figsize=(10, 7))
dendrogram(Z)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Index")
plt.ylabel("Distance")
plt.show()

# Step 3
# Part 2

# Assigning clusters using the linkage matrix Z and a maxclust criterion (here, 3 clusters)
clusters_hierarchy = fcluster(Z, t=3, criterion="maxclust")

# Adding the cluster labels to the original dataframe
df["HierCluster"] = clusters_hierarchy

# Creating a scatterplot for the two features with points colored by their cluster label
plt.figure(figsize=(8,6))
plt.scatter(data_hierarchy["Parks per 10,000 residents"],
            data_hierarchy["Acres per 1,000 people"],
            c=clusters_hierarchy, cmap="viridis", edgecolor="k", s=100)
plt.xlabel("Parks per 10,000 residents")
plt.ylabel("Acres per 1,000 people")
plt.title("Hierarchical Clustering Scatter Plot")

plt.show()

# Step 4
# Part 1

# Selecting features for k-means clustering
features_kmeans = ["Population", "investment_dollars"]
data_kmeans = df_normalized[features_kmeans]

# Running k-means with a range of cluster numbers and record the distortions (inertia)
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_kmeans)
    distortions.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(8,6))
plt.plot(K, distortions, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Distortion")
plt.title("Elbow Plot for K-Means Clustering")
plt.show()

# Step 4
# Part 2

# Performing k-means clustering using the chosen optimal number (e.g., 3)
optimal_k = 3
kmeans_model = KMeans(n_clusters=optimal_k, random_state=42)
clusters_kmeans = kmeans_model.fit_predict(data_kmeans)
df["KMeansCluster"] = clusters_kmeans

# Scattering plot for the selected features with cluster colors
plt.figure(figsize=(8,6))
plt.scatter(data_kmeans[features_kmeans[0]], data_kmeans[features_kmeans[1]], c=clusters_kmeans, cmap="plasma")
plt.xlabel(features_kmeans[0])
plt.ylabel(features_kmeans[1])
plt.title("K-Means Clustering Scatter Plot")
plt.show()

# Step 5

# Using all normalized numeric features for clustering
data_all = df_normalized.copy()

# Choosing the number of clusters (using 3 as an example)
optimal_k_all = 3
kmeans_all = KMeans(n_clusters=optimal_k_all, random_state=42)
clusters_all = kmeans_all.fit_predict(data_all)
df["AllNumericCluster"] = clusters_all

# Displaying cluster summary statistics (mean values for each cluster)
cluster_summary = df.groupby("AllNumericCluster")[numeric_cols].mean()
print("Cluster Summary (mean values):")
print(cluster_summary)

# Step 5
# Part 1

# Boxplot of distribution by cluster

plt.figure(figsize=(8,6))
df.boxplot(column="Population", by="AllNumericCluster")
plt.title("Distribution of Population by Cluster")
plt.suptitle("")
plt.xlabel("Cluster")
plt.ylabel("Population")
plt.show()

# Bar charrt: Average parks per 10,000 residents by cluster

cluster_means = df.groupby("AllNumericCluster")["Parks per 10,000 residents"].mean()
plt.figure(figsize=(8,6))
plt.bar(cluster_means.index.astype(str), cluster_means.values)
plt.xlabel("Cluster")
plt.ylabel("Average Parks per 10,000 Residents")
plt.title("Average Parks per 10,000 Residents by Cluster")
plt.show()

# Historgram of swimming pools by cluster

clusters = sorted(df["AllNumericCluster"].unique())
plt.figure(figsize=(10, 6))
for cluster in clusters:
    subset = df[df["AllNumericCluster"] == cluster]
    plt.hist(subset["Swimming_pools"], alpha=0.5, label=f"Cluster {cluster}")
plt.xlabel("Swimming Pools")
plt.ylabel("Frequency")
plt.title("Histogram of Swimming Pools by Cluster")
plt.legend()
plt.show()

"""Since we don’t have an individual “Facilities” column, I created  a new metric by summing several facility-related columns:

Facility-related columns:

"Fields/ Diamonds", "Tennis_dedicdated", "Pickleball_dedicated", "Pickleball_combined", "Hoops", "Community_garden_sites", "Dog_parks", "Playgrounds", "Rec_senior_centers", "Restrooms", "Skateparks", "Splashpads", "Swimming_pools", "Disc_golf_courses"

We then calculate Total_Facilities per 10,000 residents as:

Total Facilities per 10,000
=
Total Facilities/
Population
×
10000


"""

# Listing of columns that represent different facility counts
facility_cols = [
    "Fields/ Diamonds",
    "Tennis_dedicdated",
    "Pickleball_dedicated",
    "Pickleball_combined",
    "Hoops",
    "Community_garden_sites",
    "Dog_parks",
    "Playgrounds",
    "Rec_senior_centers",
    "Restrooms",
    "Skateparks",
    "Splashpads",
    "Swimming_pools",
    "Disc_golf_courses"
]

# Creating a new column for Total Facilities by summing all facility-related columns
df["Total_Facilities"] = df[facility_cols].sum(axis=1)

# Calculating Facilities per 10,000 residents (to normalize by population)
df["Facilities per 10,000 residents"] = (df["Total_Facilities"] / df["Population"]) * 10000

# Displaying a few rows to confirm the new columns
df[["City", "Population", "Total_Facilities", "Facilities per 10,000 residents"]].head()

# Visualizing it
# 1 scatterplot
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
scatter = plt.scatter(
    df["Population"],
    df["Facilities per 10,000 residents"],
    c=df["AllNumericCluster"],
    cmap="coolwarm",
    edgecolor="k",
    s=70
)
plt.xlabel("Population")
plt.ylabel("Facilities per 10,000 Residents")
plt.title("Facilities per 10,000 Residents vs. the recordedPopulation")
plt.colorbar(scatter, label="AllNumericCluster")
plt.show()

# Barchart

cluster_avg_facilities = df.groupby("AllNumericCluster")["Facilities per 10,000 residents"].mean()

plt.figure(figsize=(8,6))
bars = plt.bar(cluster_avg_facilities.index.astype(str),
               cluster_avg_facilities.values,
               color="skyblue", edgecolor="k")
plt.xlabel("Cluster")
plt.ylabel("Average Facilities per 10,000 Residents")
plt.title("Average Facilities per 10,000 Residents by Cluster")
plt.show()

print("Average Facilities per 10,000 residents by cluster:")
print(cluster_avg_facilities)

# Boxplot of features across all

# Defining the list of numeric features you want to explore
numeric_vars = [
    "Population",
    "Parks per 10,000 residents",
    "Acres per 1,000 people",
    "investment_dollars",
    "Facilities per 10,000 residents"
]

# Number of features
num_plots = len(numeric_vars)

# Creating subplots for each numeric feature's distribution by cluster
plt.figure(figsize=(18, 5))

for i, var in enumerate(numeric_vars, 1):
    plt.subplot(1, num_plots, i)
    df.boxplot(column=var, by="AllNumericCluster")
    plt.title(var)
    plt.xlabel("Cluster")
    plt.ylabel(var)
    # Remove the automatic subtitle that pandas adds
    plt.suptitle("")

plt.tight_layout()
plt.show()

"""The scatter plot shows that as population increases, facilities per 10,000 residents tend to drop, which is interesting however it ends up suggesting that larger cities might struggle to keep up on a per-capita basis. The bar chart then clearly highlights which clusters have, on average, higher or lower facilities per 10,000 residents, making it easy to see which groups are better served and developed. Finally, the box plots break down the distributions of key metrics—like population and park acreage—across clusters, revealing not only typical values but also the range and outliers, which gives a more detailed picture of the diversity within each single cluster.

### Step 7

Through my analysis, I discovered that the clustering techniques reveal meaningful differences among the different recordd cities: the hierarchical clustering using “Parks per 10,000 residents” and “Acres per 1,000 people” grouped cities in a way that highlighted variations in park availability and size, while the k-means clustering on “Population” and “investment_dollars” underscored differences in city scale and financial commitment that was being made to parks. Clustering on all numeric features further refined these insights, showing nuanced profiles where larger cities often have lower facilities per capita (which was an interesting finding in my opinioN). The extended exploration—with scatter, bar, and box plots of the newly derived “Facilities per 10,000 residents” metric—confirmed that some smaller cities tend to provide more recreational resources per resident compared to their larger counterparts, offering a comprehensive picture of how urban planning, investment, and resource allocation vary across the public land services dataset.

## Part 2

The question I wanted to explore is

Do cities that invest more in parks and recreation on a per capita basis tend to provide more recreational facilities per resident?

To investigate this, I first tried to created a new metric - Investment per 10k Residents - by normalizing the raw investment dollars by each city’s population. I then compared this with the already derived Facilities per 10,000 Residents metric using a scatterplot, with points colored by their previously determined cluster.

From this scatterplot, we see that cities with higher **Investment per 10k Residents** often have higher **Facilities per 10,000 Residents**, suggesting a generally positive relationship between per-capita funding and facility availability. However, there are clear outliers—some cities have relatively high investment but do not provide proportionally more facilities, while others manage more facilities despite lower investment levels. Coloring by cluster also reveals that certain groups of cities (e.g., those in the lighter-colored cluster) tend to cluster at higher investment/facility ratios, whereas darker-colored clusters typically appear in the lower-left range (indicating lower investment and fewer facilities per capita).
"""

# Calculating the new metric: Investment per 10,000 residents
df["Investment per 10k residents"] = (df["investment_dollars"] / df["Population"]) * 10000

# Scatterplot to compare Investment per 10k residents vs. Facilities per 10,000 residents
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    df["Investment per 10k residents"],
    df["Facilities per 10,000 residents"],
    c=df["AllNumericCluster"],
    cmap="viridis",
    edgecolor="k",
    s=70
)
plt.xlabel("Investment per 10k Residents")
plt.ylabel("Facilities per 10,000 Residents")
plt.title("Facilities vs. Investment per Capita in Parks & Recreation")
plt.colorbar(scatter, label="Cluster")
plt.show()


# Boxplot of Investment per 10k Residents across clusters
plt.figure(figsize=(8,6))
df.boxplot(column="Investment per 10k residents", by="AllNumericCluster")
plt.title("Investment per 10k Residents by Cluster")
plt.suptitle("")
plt.xlabel("Cluster")
plt.ylabel("Investment per 10k Residents")
plt.show()

"""Overall, while investment is a key factor influencing facilities, these outliers indicate that other factors—such as local policy choices, existing infrastructure, political fabours or demographic needs—also play a role in determining how many amenities cities provide."""
