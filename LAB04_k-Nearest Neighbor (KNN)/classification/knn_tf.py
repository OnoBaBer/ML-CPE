import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_and_find_best_k(X_train, y_train, X_test, y_test, k_values=[3, 5, 7, 9]):
    print("[2/4] กำลังเทรนโมเดล KNN เพื่อหาค่า K ที่ดีที่สุด...")
    
    # สร้างโฟลเดอร์เก็บผลลัพธ์
    os.makedirs('outputs', exist_ok=True)
    
    accuracy_scores = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracy_scores.append(acc)
        print(f"      -> ทดสอบ k={k}: ความแม่นยำ = {acc:.4f}")
        
    best_k = k_values[np.argmax(accuracy_scores)]
    best_acc = max(accuracy_scores)
    print(f"      => ค่า K ที่ดีที่สุดคือ k={best_k} (ความแม่นยำ: {best_acc:.4f})")
    
    # พล็อตและบันทึกกราฟ K-Curve
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, accuracy_scores, marker='o', linestyle='dashed', color='b')
    plt.title('Accuracy vs. K Value')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig('outputs/01_k_curve.png')
    plt.close()
    
    # เทรนโมเดลตัวสมบูรณ์ด้วยค่า K ที่ดีที่สุด
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    
    return best_knn, best_k