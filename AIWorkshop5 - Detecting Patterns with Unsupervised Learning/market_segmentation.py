import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from sklearn.preprocessing import StandardScaler

# 1. Load and preprocess the sales data (from 'sales.csv')
# Handle potential missing values and extract valid sales features
# Assume 'sales.csv' contains columns: StoreID, Tshirt, TankTop, HalterTop, Turtleneck, TubeTop, Sweater
sales_data = pd.read_csv('sales.csv')

# Step 1.1: Clean data (fill missing values and remove invalid rows)
# Fill missing sales values with 0 (reasonable for "no units sold")
sales_data_clean = sales_data.fillna(0)
# Remove rows with non-numeric sales values or negative sales (invalid data)
sales_features = sales_data_clean[['Tshirt', 'Tank top', 'Halter top', 'Turtleneck', 'Tube top', 'Sweater']]
sales_features = sales_features.apply(pd.to_numeric, errors='coerce').fillna(0)  # Ensure numeric type
sales_features = sales_features[sales_features >= 0].dropna()  # Remove rows with negative values

# Step 1.2: Standardize features (reduce impact of large value differences, improves Mean Shift performance)
scaler = StandardScaler()
sales_features_scaled = scaler.fit_transform(sales_features)

# Print preprocessing results for verification
print(f"Shape of cleaned & scaled sales features: {sales_features_scaled.shape}")
print("First 5 rows of cleaned & scaled sales features:\n", np.round(sales_features_scaled[:5], 2))

# 2. Estimate optimal bandwidth for Mean Shift (critical parameter for cluster quality)
# Quantile controls bandwidth: smaller quantile → smaller bandwidth → more clusters; larger quantile → fewer clusters
# Use 10% of data for bandwidth estimation (balances accuracy and efficiency)
bandwidth = estimate_bandwidth(sales_features_scaled, quantile=0.1, n_samples=int(len(sales_features_scaled)*0.1))
print(f"\nEstimated bandwidth for Mean Shift: {bandwidth:.4f}")

# 3. Train Mean Shift clustering model (core algorithm for market segmentation)
meanshift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=-1)  # n_jobs=-1 uses all CPU cores
meanshift_model.fit(sales_features_scaled)

# 4. Extract clustering results (key outputs for market analysis)
cluster_centers_scaled = meanshift_model.cluster_centers_  # Scaled cluster centers
labels = meanshift_model.labels_  # Cluster label for each store
num_clusters = len(np.unique(labels))  # Total number of market segments

# Inverse scale cluster centers to original sales units (for intuitive interpretation)
cluster_centers_original = scaler.inverse_transform(cluster_centers_scaled)

# 5. Organize and print results (align with market segmentation analysis needs)
print(f"\nTotal number of market segments (clusters): {num_clusters}")

# Create DataFrame for cluster centers (original sales units)
cluster_centers_df = pd.DataFrame(
    np.round(cluster_centers_original).astype(int),  # Round to integer (sales units are whole numbers)
    columns=['Tshirt', 'Tank top', 'Halter top', 'Turtleneck', 'Tube top', 'Sweater'],
    index=[f'Cluster {i+1}' for i in range(num_clusters)]
)
print("\nAverage sales volume per product in each market segment (units):")
print(cluster_centers_df)

# Count number of stores in each cluster (segment size)
cluster_store_count = pd.Series(labels).value_counts().sort_index()
cluster_store_count.index = [f'Cluster {i+1}' for i in cluster_store_count.index]
print("\nNumber of stores in each market segment:")
print(cluster_store_count)

# 6. Visualize clustering results (2D visualization for intuitive understanding)
# Use PCA to reduce 6 features to 2 dimensions (for plotting; preserves key variance)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
sales_features_pca = pca.fit_transform(sales_features_scaled)
cluster_centers_pca = pca.transform(cluster_centers_scaled)

# Plot settings (clear and professional for analysis reports)
plt.figure(figsize=(12, 8))
markers = cycle(['o', 's', '^', 'D', 'x', '*', 'p', 'v'])  # Different markers for each cluster
colors = plt.cm.Set3(np.linspace(0, 1, num_clusters))  # Distinct colors for each cluster

# Plot each cluster's stores and center
for cluster_id, marker, color in zip(range(num_clusters), markers, colors):
    # Plot stores in the current cluster
    cluster_mask = labels == cluster_id
    plt.scatter(
        sales_features_pca[cluster_mask, 0],
        sales_features_pca[cluster_mask, 1],
        marker=marker,
        color=color,
        s=80,
        alpha=0.7,
        label=f'Cluster {cluster_id+1} (Stores: {cluster_store_count[f"Cluster {cluster_id+1}"]})'
    )
    # Plot cluster center (highlighted for clarity)
    plt.scatter(
        cluster_centers_pca[cluster_id, 0],
        cluster_centers_pca[cluster_id, 1],
        marker='o',
        color='black',
        s=200,
        edgecolor='white',
        linewidth=2,
        label=f'Cluster {cluster_id+1} Center' if cluster_id == 0 else ""  # Avoid duplicate center labels
    )

# Add plot labels and legend (informative for stakeholders)
plt.title('Market Segmentation of Stores Based on Shopping Patterns (Mean Shift)', fontsize=16, pad=20)
plt.xlabel(f'PCA Dimension 1 (Explains {np.round(pca.explained_variance_ratio_[0]*100, 1)}% Variance)', fontsize=12)
plt.ylabel(f'PCA Dimension 2 (Explains {np.round(pca.explained_variance_ratio_[1]*100, 1)}% Variance)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()  # Adjust layout to prevent label cutoff

# 7. Save results (for project submission and further analysis)
# Save cluster centers and store counts to CSV (structured data for reports)
cluster_centers_df.to_csv('market_segment_cluster_centers.csv', index=True)
cluster_store_count.to_csv('market_segment_store_count.csv', index=True, header=['Store Count'])
# Save visualization to PNG (high-resolution for presentations)
plt.savefig('market_segmentation_visualization.png', dpi=300, bbox_inches='tight')

print("\nResults saved successfully:")
print("- Cluster centers: market_segment_cluster_centers.csv")
print("- Store count per segment: market_segment_store_count.csv")
print("- Visualization: market_segmentation_visualization.png")