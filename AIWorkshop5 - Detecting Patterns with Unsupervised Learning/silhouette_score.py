import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

# Load data from input file
X_original = np.loadtxt('data_quality.txt', delimiter=',')

x_min, x_max = X_original[:, 0].min() - 1, X_original[:, 0].max() + 1
y_min, y_max = X_original[:, 1].min() - 1, X_original[:, 1].max() + 1

random_data = np.column_stack([
    np.random.uniform(x_min, x_max, 100), 
    np.random.uniform(y_min, y_max, 100)   
])
X_new = np.vstack((X_original, random_data))
print(X_new.shape)  # 确认100个随机数已添加（总样本数增加100）


# Plot input data
plt.figure()
plt.scatter(X_new[:,0], X_new[:,1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X_new[:, 0].min() - 1, X_new[:, 0].max() + 1
y_min, y_max = X_new[:, 1].min() - 1, X_new[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Initialize variables
scores = []
values = np.arange(2, 10)

# Iterate through the defined range
for num_clusters in values:
    # Train the KMeans clustering model
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X_new)

    score = metrics.silhouette_score(X_new, kmeans.labels_, 
                    metric='euclidean', sample_size=len(X_new))

    print("\nNumber of clusters =", num_clusters)
    print("Silhouette score =", score)
    scores.append(score)

# Plot silhouette scores
plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('Silhouette score vs number of clusters')

# Extract best score and optimal number of clusters
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters =', num_clusters)

plt.show()