import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Load dataset
df = pd.read_csv("mall_customers.csv")
# Select features
X = df[['Annual Income', 'Spending Score']]
# Apply KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
# Predict clusters
df['Cluster'] = kmeans.labels_
# Plot clusters
plt.scatter(df['Annual Income'], df['Spending Score'], c=df['Cluster'], cmap='viridis')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()
