import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def visualize_and_save(X_pca, kmeans_model, original_df):
    print("[5/5] กำลังสร้างภาพนิทัศน์และบันทึกไฟล์ Outputs...")
    os.makedirs('outputs', exist_ok=True)
    
    labels = kmeans_model.labels_
    
    # 1. วาดกราฟ 02_clusters.png
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='viridis', s=50, alpha=0.6)
    plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:, 1], 
                c='red', marker='X', s=200, label='Centroids')
    plt.title('K-Means Clustering Results (PCA 2D)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.savefig('outputs/02_clusters.png')
    plt.close()
    
    # นำผลลัพธ์การจัดกลุ่มไปใส่เพิ่มใน DataFrame ต้นฉบับ
    original_df_clustered = original_df.copy()
    original_df_clustered['Cluster'] = labels
    
    # 2. สร้าง cluster_summary.csv (สรุปค่าเฉลี่ยของข้อมูลตัวเลขในแต่ละกลุ่ม)
    num_cols = original_df_clustered.select_dtypes(include=['int64', 'float64']).columns
    cluster_summary = original_df_clustered.groupby('Cluster')[num_cols].mean()
    cluster_summary.to_csv('outputs/cluster_summary.csv')
    
    # 3. สร้าง clustered_data.csv (ปรับชื่อไฟล์จากสัตว์เป็น data ให้เข้ากับชุดข้อมูล)
    original_df_clustered.to_csv('outputs/clustered_data.csv', index=False)
    
    print("เสร็จสมบูรณ์! ตรวจสอบไฟล์ทั้งหมดได้ที่โฟลเดอร์ 'outputs/'")