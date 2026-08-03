from data_loader import load_and_preprocess_for_clustering
from kmeans_tf import find_optimal_k_and_cluster
from knn_tools import find_nearest_to_centroids
from visualize import visualize_and_save
def main():
    print("========== Start==========")
    original_df, X_pca = load_and_preprocess_for_clustering()
    kmeans_model, optimal_k = find_optimal_k_and_cluster(X_pca, max_k=10)
    representatives = find_nearest_to_centroids(X_pca, kmeans_model, original_df)
    print("\n[Example data] Middle points of each cluster:")
    print(representatives[['Cluster_ID', 'age', 'hours-per-week', 'income']])
    print("")
    visualize_and_save(X_pca, kmeans_model, original_df)
    print("========== Finished ==========")
if __name__ == "__main__":
    main()  