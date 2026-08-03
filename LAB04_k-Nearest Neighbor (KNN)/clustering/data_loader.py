import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_and_preprocess_for_clustering(filepath='adult.csv'):    
    print("[1/5] กำลังโหลดและเตรียมข้อมูลสำหรับ Clustering...")
    
    # โหลดชุดข้อมูล
    df = pd.read_csv(filepath)
    
    # กำจัดช่องว่างที่ติดมากับข้อความ และลบ Missing Values
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    
    # เลือกเฉพาะคอลัมน์ที่เป็นตัวเลขสำหรับการจัดกลุ่มด้วย K-Means
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X_num = df[num_cols]
    
    # ทำ Feature Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    
    # ทำ PCA ลดมิติข้อมูลเหลือ 2 มิติ (เพื่อให้พล็อตจุด x, y บนกราฟได้)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"เตรียมข้อมูลเสร็จสิ้น! (ใช้ตัวเลข {len(num_cols)} คอลัมน์ และลดมิติเหลือ 2D)")
    return df, X_pca