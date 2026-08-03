from data_loader import load_and_preprocess_data
from knn_tf import train_and_find_best_k
from evaluate import evaluate_and_save

def main():
    print("========== เริ่มต้นการทำงาน KNN Classification Pipeline ==========")
    
    # 1. โหลดข้อมูล (นำไฟล์ adult.csv มาวางไว้ในโฟลเดอร์เดียวกับ main.py)
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # 2. ค้นหา K ที่ดีที่สุด และสร้างกราฟ K-Curve (ทดลองที่ k=3, 5, 7, 9, 11)
    best_model, best_k = train_and_find_best_k(X_train, y_train, X_test, y_test, k_values=[3, 5, 7, 9, 11])
    
    # 3. ประเมินผล สร้างกราฟ Matrix และบันทึกไฟล์ CSV
    evaluate_and_save(best_model, X_test, y_test, best_k)
    
    print("========== ทำงานเสร็จสมบูรณ์ ==========")

if __name__ == "__main__":
    main()