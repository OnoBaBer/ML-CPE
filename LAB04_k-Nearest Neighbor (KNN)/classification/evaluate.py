import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_and_save(model, X_test, y_test, best_k):
    print(f"[3/4] กำลังประเมินผลโมเดล (k={best_k}) และสร้างไฟล์ Outputs...")
    os.makedirs('outputs', exist_ok=True)
    
    # ทำนายผลบนชุดทดสอบ
    y_pred = model.predict(X_test)
    
    # พิมพ์รายงานสรุปผล
    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
    
    # พล็อตและบันทึก Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.title(f'Confusion Matrix (KNN, k={best_k})')
    plt.xlabel('Predicted Income')
    plt.ylabel('Actual Income')
    plt.savefig('outputs/02_confusion_matrix.png')
    plt.close()
    
    # บันทึกไฟล์พยากรณ์
    print("[4/4] กำลังบันทึกผลการทำนายลงไฟล์ CSV...")
    predictions_df = pd.DataFrame({
        'Actual_Income': y_test.values,
        'Predicted_Income': y_pred
    })
    
    # แปลงเลข 0, 1 กลับเป็นข้อความเพื่อให้ผู้ใช้อ่านง่าย
    predictions_df['Actual_Income'] = predictions_df['Actual_Income'].map({0: '<=50K', 1: '>50K'})
    predictions_df['Predicted_Income'] = predictions_df['Predicted_Income'].map({0: '<=50K', 1: '>50K'})
    
    predictions_df.to_csv('outputs/predictions.csv', index=False)
    print("เสร็จสมบูรณ์! คุณสามารถตรวจสอบผลลัพธ์ได้ในโฟลเดอร์ 'outputs/'")