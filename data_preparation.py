import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np
import time
import plot as plt
import data_selection as ds

##########################################
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
        file_path = 'Blind_data_classifier/bases/05.csv'
        return pd.read_csv(file_path, header=None)

##########################################
def preprocess_data(df):
    """
    Preprocess the imported data.

    Parameters:
    df (pd.DataFrame): DataFrame containing the imported data.

    Returns:
    pd.DataFrame: DataFrame containing the preprocessed data.
    """
    df_processed = df.copy()

    #Handle missing values
    df_processed = df_processed.dropna()

    #Handle duplicates: Remove duplicate rows
    initial_rows = df_processed.shape[0]
    df_processed.drop_duplicates(inplace=True)
    if df_processed.shape[0] < initial_rows:
        print(f"  Removed {initial_rows - df_processed.shape[0]} duplicate rows.")

    # IDENTIFICAR E SEPARAR A COLUNA ALVO TEMPORARIAMENTE ANTES DA NORMALIZAÇÃO
    # Assumindo que a última coluna é sempre a coluna alvo.
    # Se não for, você precisará de uma forma mais robusta de identificá-la.
    target_column_name = df_processed.columns[-1]
    
    # Selecionar apenas as colunas numéricas que NÃO são o target para normalização
    numerical_features_cols = df_processed.drop(columns=[target_column_name]).select_dtypes(include=[np.number]).columns
    
    if not numerical_features_cols.empty:
        std_dev = df_processed[numerical_features_cols].std()
        #Avoid division by zero in case std_dev is zero
        std_dev = std_dev.replace(0, 1)
        df_processed[numerical_features_cols] = (df_processed[numerical_features_cols] - df_processed[numerical_features_cols].mean()) / std_dev

    return df_processed

##########################################
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

##########################################
def rfe_selection(X, y, n_features_to_select=100):
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

##########################################
def save_processed_data(df):
    """
    Save the processed DataFrame to a CSV file.
    
    Parameters:
    df (DataFrame): The DataFrame to save.
    """
    output_path = 'Blind_data_classifier/processed_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

##########################################
def process_data(file_path=None, plot=False, save=False, n_features_to_select=10, first_execution=False):
    if first_execution: file_path = ds.find_best(file_path,plot_results=plot)

    print("Starting data processing...")
    inicio = time.time()
    df = import_data(file_path)
    df = preprocess_data(df)
    features = rfe_selection(df.drop(columns=df.columns[-1]), df[df.columns[-1]], n_features_to_select=n_features_to_select)
    df = df[features + [df.columns[-1]]]  #Keep only selected features
    if save: save_processed_data(df)
    fim = time.time()

    print(f"Data processing completed in {fim - inicio:.2f} seconds.")
    xtrain, xtest, ytrain, ytest = split_data(df, target_column=df.columns[-1])
    if plot:
        plt.plot_data_distribution(df, target_column=df.columns[-1])
        plt.plot_features_selection(df, features)
        plt.plot_train_test_data(xtrain, ytrain, xtest, ytest, title='Train and Test Data Distribution')
    return (xtrain, xtest, ytrain, ytest)
