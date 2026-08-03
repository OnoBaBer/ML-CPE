from sklearn.neighbors import NearestNeighbors
import pandas as pd
def find_nearest_to_centroids(X_pca, kmeans_model, original_df):
    print("[4/5] Finding representative points closest to each cluster centroid")
    centroids = kmeans_model.cluster_centers_
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_pca)
    distances, indices = nn.kneighbors(centroids)
    representatives = original_df.iloc[indices.flatten()].copy()
    representatives.insert(0, 'Cluster_ID', range(len(centroids)))
    return representatives