# 📝 LAB04: Machine Learning - KNN & Clustering

```text
**[1] โครงสร้างโฟลเดอร์ของโปรเจกต์ (Project Structure)**
    LAB04/
    ├── data-adult-income/
    │   └── adult.csv                      # ชุดข้อมูลต้นฉบับ (Adult Dataset)
    │
    ├── classification/                    # ส่วนที่ 1: K-Nearest Neighbors (Classification)
    │   ├── data_loader.py                 # โหลดและทำ Data Preprocessing
    │   ├── knn_tf.py                      # โมเดล KNN และการหาค่า K ที่เหมาะสม
    │   ├── evaluate.py                    # ประเมินผลโมเดล (Accuracy, Confusion Matrix)
    │   ├── main.py                        # ไฟล์หลักสำหรับสั่งรัน Classification Pipeline
    │   └── outputs/                       # โฟลเดอร์เก็บผลลัพธ์สาย Classification
    │       ├── 01_k_curve.png             # กราฟเปรียบเทียบค่า Accuracy ในแต่ละค่า K
    │       ├── 02_confusion_matrix.png    # ภาพ Confusion Matrix
    │       └── predictions.csv            # ไฟล์บันทึกผลการทำนายข้อมูล
    │
    ├── clustering/                        # ส่วนที่ 2: K-Means & KNN Centroid (Clustering)
    │   ├── data_loader.py                 # โหลดข้อมูล ทำ Scaling และลดมิติด้วย PCA
    │   ├── kmeans_tf.py                   # คำนวณ Elbow Method และจัดกลุ่มด้วย K-Means
    │   ├── knn_tools.py                   # ใช้ KNN หาตัวแทนข้อมูลที่ใกล้ Centroid มากที่สุด
    │   ├── visualize.py                   # สร้างกราฟการกระจายตัวและสรุปผล
    │   ├── main.py                        # ไฟล์หลักสำหรับสั่งรัน Clustering Pipeline
    │   └── outputs/                       # โฟลเดอร์เก็บผลลัพธ์สาย Clustering
    │       ├── 01_elbow.png               # กราฟ Elbow Method เลือกค่า K
    │       ├── 02_clusters.png            # กราฟ Scatter Plot การจัดกลุ่มแบบ PCA 2D
    │       ├── cluster_summary.csv        # ตารางสรุปค่าเฉลี่ยของแต่ละกลุ่ม
    │       └── clustered_data.csv         # ชุดข้อมูลพร้อมคอลัมน์บอกกลุ่ม (Cluster ID)
    │
    ├── requirements.txt                   # รายชื่อ Python Libraries ที่จำเป็น
    └── link-data.txt                      # ลิงก์และรายละเอียดแหล่งที่มาของข้อมูล

[2] การติดตั้งไลบรารีที่จำเป็น (Installation)
    เปิด Terminal/PowerShell ที่หน้า Root Directory ของโปรเจกต์ แล้วใช้คำสั่ง:

    pip install -r requirements.txt

    วิธีการรันโปรแกรม (How to Run)
        --- การรันส่วนที่ 1: Classification ---

        เปลี่ยนตำแหน่งโฟลเดอร์ไปยังโฟลเดอร์ classification:

        cd classification

        สั่งรันไฟล์ main.py:

        python main.py

        ผลลัพธ์ทั้งหมดจะถูกสร้างขึ้นอัตโนมัติในโฟลเดอร์ classification/outputs/

    --- การรันส่วนที่ 2: Clustering ---

        สลับโฟลเดอร์ไปยัง clustering:

        cd ../clustering

        สั่งรันไฟล์ main.py:

        python main.py

        ผลลัพธ์ทั้งหมดจะถูกสร้างขึ้นอัตโนมัติในโฟลเดอร์ clustering/outputs/