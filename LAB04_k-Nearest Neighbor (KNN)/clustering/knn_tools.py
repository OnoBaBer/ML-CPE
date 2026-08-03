from sklearn.neighbors import NearestNeighbors
import pandas as pd

def find_nearest_to_centroids(X_pca, kmeans_model, original_df):
    print("[4/5] กำลังใช้ KNN ค้นหาตัวแทนที่อยู่ใกล้จุด Centroid ของแต่ละกลุ่ม...")
    
    # จุดกึ่งกลางของแต่ละกลุ่ม
    centroids = kmeans_model.cluster_centers_
    
    # ใช้ KNN หา 1 จุดที่ใกล้ Centroid มากที่สุด
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(X_pca)
    
    distances, indices = nn.kneighbors(centroids)
    
    # ดึงข้อมูลของบุคคลที่เป็นตัวแทนของกลุ่มจาก DataFrame ต้นฉบับ
    representatives = original_df.iloc[indices.flatten()].copy()
    representatives.insert(0, 'Cluster_ID', range(len(centroids)))
    
    return representatives