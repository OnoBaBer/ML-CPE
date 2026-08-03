import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def load_and_preprocess_data(filepath='adult.csv'):
    print('[1/4] Loading and preprocessing data for Classification')
    df = pd.read_csv(filepath)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    X = df.drop(columns=['income'])
    y = df['income']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f"Finished! amount of data : {X_train_scaled.shape[0]} row")
    return X_train_scaled, X_test_scaled, y_train, y_test