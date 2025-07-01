import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def import_data(file_path=None):
    if file_path:
        return pd.read_csv(file_path, header=None)
    else: 
        file_path = '/home/jvfg/Documents/SI/Algoritmos de Classificação/bases/05.csv'
    return pd.DataFrame()

def preprocess_data(df):
    #Basic preprocessing steps
    df = df.dropna()
    df = df.drop_duplicates()
    return df

def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)
