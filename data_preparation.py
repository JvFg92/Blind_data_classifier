import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np

def import_data(file_path=None):
    """Import data from a CSV file.
    Parameters:
    file_path (str): Path to the CSV file. If None, a default path is used.
    Returns:
    DataFrame: A pandas DataFrame containing the imported data.
    If file_path is None, it defaults to a predefined path.
    """
    if file_path:
        return pd.read_csv(file_path, header=None)
    else: 
        file_path = '/home/jvfg/Documents/SI/Algoritmos de Classificação/bases/05.csv'
        return pd.read_csv(file_path, header=None)


def preprocess_data(df):
    """
    Preprocess the imported data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the imported data.

    Returns:
    pd.DataFrame: DataFrame containing the preprocessed data.
    """
    df_processed = df.copy()

    # Handle missing values
    df_processed = df_processed.dropna()

    # Handle duplicates: Remove duplicate rows
    initial_rows = df_processed.shape[0]
    df_processed.drop_duplicates(inplace=True)
    if df_processed.shape[0] < initial_rows:
        print(f"  Removed {initial_rows - df_processed.shape[0]} duplicate rows.")

    # Normalize numerical features
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    if not numerical_cols.empty:
        std_dev = df_processed[numerical_cols].std()
        # Avoid division by zero in case std_dev is zero
        std_dev = std_dev.replace(0, 1)
        df_processed[numerical_cols] = (df_processed[numerical_cols] - df_processed[numerical_cols].mean()) / std_dev

    return df_processed

def split_data(df, target_column):
    """
    Split the dataset into training and testing sets.
    Parameters:
    df (DataFrame): The input DataFrame containing features and target.
    target_column (str): The name of the target column in the DataFrame.
    Returns:
    X_train, X_test, y_train, y_test: Split datasets for training and testing
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def rfe_selection(X, y, n_features_to_select=10):
    """
    Perform Recursive Feature Elimination (RFE) to select the top features.
    
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    n_features_to_select (int): Number of top features to select.
    
    Returns:
    selected_features (list): List of selected feature names.
    """
    
    model = LogisticRegression(max_iter=1000)
    rfe = RFE(model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    
    selected_features = X.columns[rfe.support_].tolist()
    return selected_features