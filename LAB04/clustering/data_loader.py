import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
def load_and_preprocess_for_clustering(filepath='../data-adult-income/adult.csv'): 
    print("[1/5] Loading and preprocessing data for Clustering")
    df = pd.read_csv(filepath)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    X_num = df[num_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    print(f"Finished preprocessing data for clustering (Using {len(num_cols)} numerical columns and reducing dimensions to 2D)")
    return df, X_pca