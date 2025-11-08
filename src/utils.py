# utils.py
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def read_csv(csv_path):
    """
    Reads CSV and ensures correct columns exist.
    Expected columns: ['id_code', 'diagnosis']
    """
    df = pd.read_csv(csv_path)
    if 'id_code' not in df.columns or 'diagnosis' not in df.columns:
        raise ValueError(f"CSV {csv_path} must have 'id_code' and 'diagnosis' columns.")
    return df

def get_class_weights(df):
    """
    Computes class weights to fix imbalance
    """
    y = df['diagnosis'].values
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))
