# Regression & Classification on HR Dataset

* **ชุดข้อมูลที่ใช้อ้างอิง**

ชุดข้อมูล HRDataset_v14 จากระบบข้อมูลทรัพยากรบุคคล (Human Resources) 

**Kaggle** : https://www.kaggle.com/datasets/rhuebner/human-resources-data-set

* **วัตถุประสงค์**
1. ศึกษาหลักการทำงานของ Regression และ Classification 
2. ประยุกต์ใช้ Linear Regression ทำนายอายุ (Age) ของพนักงาน โดยแปลงจากวันเกิด (DOB) 
3. ประยุกต์ใช้ Logistic Regression เพื่อจำแนกเพศ (GenderID) ของพนักงาน 
4. ฝึกการเตรียมข้อมูล (Feature Scaling) และประยุกต์ใช้เทคนิค Principal Component Analysis (PCA) เพื่อลดจำนวนมิติข้อมูล
5. ประเมินประสิทธิภาพเปรียบเทียบโมเดลด้วยตัวชี้วัด (MAE, RMSE, R², Accuracy, Confusion Matrix, ROC Curve)

* **ขั้นตอนการทำงาน**

**1. Data Preparation & EDA**
    - นำเข้าไลบรารี `pandas`,`numpy`,`matplotlib`,`seaborn` และ `scikit-learn`
    - ดึงข้อมูล `HRDataset_v14.csv` มาวิเคราะห์และแปลงคอลัมน์ `DOB` (วันเกิด) ให้เป็นคอลัมน์ `Age` (อายุ)
    - ตรวจสอบและกำจัด Missing Values พร้อมทั้งพล็อตกราฟการกระจายตัวของอายุพนักงาน

**2. Linear Regression (ทำนายอายุพนักงาน)**
    - การทำ Simple Linear Regression โดยใช้ 1 Feature คือ `Salary` (เงินเดือน) ทำนายอายุ
    - การทำ Multiple Linear Regression โดยใช้หลาย Feature ได้แก่ เงินเดือน, คะแนนความพึงพอใจ, การขาดงาน ฯลฯ มาพยากรณ์อายุร่วมกัน
    - วัดผลโมเดลด้วยค่าความคลาดเคลื่อน MAE, RMSE และสัมประสิทธิ์การตัดสินใจ (R²)

**3. Classification & PCA (จำแนกเพศพนักงาน)**
    - กำหนดให้ `GenderID` (เพศหญิง=0, เพศชาย=1) เป็นเป้าหมาย (Target)
    - ปรับสเกลข้อมูลให้เป็นมาตรฐานด้วย `StandardScaler()`
    - ใช้หลักการลดมิติข้อมูล PCA (Principal Component Analysis) ให้เหลือ 2 องค์ประกอบหลัก เพื่อให้โมเดลมีประสิทธิภาพและลดความซับซ้อน
    - สร้างและเทรนโมเดลด้วย `LogisticRegression()`
    - วัดประสิทธิภาพด้วย Accuracy, Classification Report, Confusion Matrix และกราฟ ROC Curve (เพื่อดูพื้นที่ใต้กราฟ AUC)

**4. Model Comparison**
    - ตรวจสอบประสิทธิภาพเทียบกันระหว่างข้อมูล Train และ Test ว่ามีอาการ Overfitting หรือ Underfitting หรือไม่
    - เปรียบเทียบ MAE, RMSE และ R² ระหว่าง Simple กับ Multiple Regression
    - สรุปรวบยอดเมทริกซ์การประเมิน (Performance Metrics) ของทุกโมเดลออกมาเป็นตาราง

* **สรุปผล**
    - Regression: พบว่าโมเดลทั้งแบบ Simple และ Multiple Regression สามารถทำนายอายุของพนักงานได้ในระดับหนึ่ง แต่การใช้ Feature ที่มากขึ้นในตัว Multiple Linear Regression ช่วยกระจายน้ำหนักความสำคัญ ทำให้มีแนวโน้มปรับปรุงความแม่นยำได้ดีกว่าการใช้แค่เงินเดือนอย่างเดียว
    - การ Classification โดยการใช้ Logistic Regression ร่วมกับการลดมิติด้วย PCA สามารถจำแนกกลุ่มพนักงานชาย/หญิงได้เบื้องต้น โดยสามารถดูผลลัพธ์การทายถูก/ผิด อย่างละเอียดได้จาก Confusion Matrix
    - โดยมีข้อสรุปทั้งหมดว่า ข้อมูลเชิงลึกด้านบุคลากร (HR) เป็นข้อมูลพฤติกรรมซึ่งอาจมีความซับซ้อนมากกว่าข้อมูลกายภาพ การทำ Feature Engineering ที่ดี (เช่น การหาอายุจากวันเกิด) และการใช้ PCA ช่วยให้สามารถนำข้อมูลมาปรับใช้เข้ากับงานวิจัยด้าน Machine Learning ได้อย่างตรงจุดประสงค์