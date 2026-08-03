import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
def find_optimal_k_and_cluster(X_pca, max_k=10):
    print("[2/5] Finding optimal K value using Elbow Method")
    os.makedirs('outputs', exist_ok=True)
    inertia = []
    k_range = range(1, max_k + 1)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)
        inertia.append(kmeans.inertia_)
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.savefig('outputs/01_elbow.png')
    plt.close()
    optimal_k = 3
    print(f"      => Selected K={optimal_k}")
    print("[3/5] Creating and training K-Means model")
    best_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    best_kmeans.fit(X_pca)
    return best_kmeans, optimal_k