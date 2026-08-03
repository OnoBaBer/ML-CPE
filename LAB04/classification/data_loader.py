import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(filepath='adult.csv'):
    print("[1/4] กำลังโหลดและเตรียมข้อมูล...")
    
    # โหลดชุดข้อมูล
    df = pd.read_csv(filepath)
    
    # กำจัดช่องว่างที่ติดมากับข้อความ และจัดการเครื่องหมาย '?'
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    
    # แปลงคอลัมน์คำตอบ (Target)
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    
    X = df.drop(columns=['income'])
    y = df['income']
    
    # แปลงข้อมูลตัวอักษรให้เป็นตัวเลขด้วย One-Hot Encoding
    X = pd.get_dummies(X, drop_first=True)
    
    # แบ่งข้อมูล Train 70% และ Test 30%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # ทำ Feature Scaling (สำคัญมากสำหรับ KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"เตรียมข้อมูลเสร็จสิ้น! จำนวนข้อมูล Training: {X_train_scaled.shape[0]} แถว")
    return X_train_scaled, X_test_scaled, y_train, y_test