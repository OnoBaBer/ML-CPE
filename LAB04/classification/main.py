from data_loader import load_and_preprocess_data
from knn_tf import train_and_find_best_k
from evaluate import evaluate_and_save
def main():
    print("========== Start ==========")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    best_model, best_k = train_and_find_best_k(X_train, y_train, X_test, y_test, k_values=[3, 5, 7, 9, 11])
    evaluate_and_save(best_model, X_test, y_test, best_k)
    print("========== Finished ==========")
if __name__ == "__main__":
    main()