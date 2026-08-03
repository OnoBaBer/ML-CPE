from data_loader import load_and_preprocess_for_clustering
from kmeans_tf import find_optimal_k_and_cluster
from knn_tools import find_nearest_to_centroids
from visualize import visualize_and_save

def main():
    print("========== เริ่มต้นการทำงาน Clustering Pipeline ==========")
    
    # 1. โหลดข้อมูล (เรียกใช้ path จากโฟลเดอร์ data_adult_income เหมือนเดิม)
    original_df, X_pca = load_and_preprocess_for_clustering()
    
    # 2. หาวิธีแบ่งกลุ่มที่ดีที่สุดและเทรน K-Means
    kmeans_model, optimal_k = find_optimal_k_and_cluster(X_pca, max_k=10)
    
    # 3. ค้นหาตัวแทนกลุ่มด้วย KNN
    representatives = find_nearest_to_centroids(X_pca, kmeans_model, original_df)
    print("\n[ข้อมูลตัวแทน] บุคคลที่อยู่ตรงกลางสุดของแต่ละกลุ่ม:")
    print(representatives[['Cluster_ID', 'age', 'hours-per-week', 'income']])
    print("")
    
    # 4. สร้างรูปภาพและบันทึกไฟล์ 
    visualize_and_save(X_pca, kmeans_model, original_df)
    
    print("========== ทำงานเสร็จสมบูรณ์ ==========")

if __name__ == "__main__":
    main()  