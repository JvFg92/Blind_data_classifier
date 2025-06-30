import pandas as pd
from sklearn.model_selection import train_test_split

def import_data(file_path):
    """
    Import data from a CSV file.
    
    Parameters:
    file_path (str): Path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the imported data.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the imported data.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the imported data.
    
    Returns:
    pd.DataFrame: DataFrame containing the preprocessed data.
    """
    # Handle missing values
    df = df.dropna()

    # Normalize numerical features
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()

    return df

def split_data(df, target_column, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the preprocessed data.
    target_column (str): Name of the target column.
    test_size (float): Proportion of the dataset to include in the test split.
    random_state (int): Random seed for reproducibility.
    
    Returns:
    tuple: Training and testing sets (X_train, X_test, y_train, y_test).
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test
