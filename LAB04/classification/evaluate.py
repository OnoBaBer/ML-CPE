import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
def evaluate_and_save(model, X_test, y_test, best_k):
    print(f"[3/4] (k={best_k}) building confusion matrix")
    os.makedirs('outputs', exist_ok=True)
    y_pred = model.predict(X_test)
    print(f"\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, target_names=['<=50K', '>50K']))
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['<=50K', '>50K'], yticklabels=['<=50K', '>50K'])
    plt.title(f'Confusion Matrix (KNN, k={best_k})')
    plt.xlabel('Predicted Income')
    plt.ylabel('Actual Income')
    plt.savefig('outputs/02_confusion_matrix.png')
    plt.close()
    print("[4/4] saving in CSV")
    predictions_df = pd.DataFrame({
        'Actual_Income': y_test.values,
        'Predicted_Income': y_pred
    })
    predictions_df['Actual_Income'] = predictions_df['Actual_Income'].map({0: '<=50K', 1: '>50K'})
    predictions_df['Predicted_Income'] = predictions_df['Predicted_Income'].map({0: '<=50K', 1: '>50K'})
    predictions_df.to_csv('outputs/predictions.csv', index=False)
    print("Finished saving to 'outputs/'")