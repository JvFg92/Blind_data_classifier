import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import time

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
        file_path = '/home/jvfg/Documents/SI/Algoritmos de Classificação/Codes/bases/05.csv'
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

def plot_data_distribution(df, target_column):
    """
    Plot the distribution of the target variable.
    
    Parameters:
    df (DataFrame): The input DataFrame containing features and target.
    target_column (str): The name of the target column in the DataFrame.
    """
    plt.figure(figsize=(10, 6))
    df[target_column].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {target_column}')
    plt.xlabel(target_column)
    plt.ylabel('Frequency')
    plt.show()

def plot_features_selection(df, selected_features):
    """
    Plot the selected features from the DataFrame.
    
    Parameters:
    df (DataFrame): The input DataFrame containing features and target.
    selected_features (list): List of selected feature names.
    """
    plt.figure(figsize=(10, 6))
    for feature in selected_features:
        plt.hist(df[feature], bins=30, alpha=0.5, label=feature)
    plt.title('Selected Features Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_data_selection(X, y):
    """
    Plot the selected features from the DataFrame.
    
    Parameters:
    X (DataFrame): The input DataFrame containing features.
    y (Series): The target variable.
    """
    plt.figure(figsize=(10, 6))
    for column in X.columns:
        plt.hist(X[column], bins=30, alpha=0.5, label=column)
    plt.title('Selected Data Distribution')
    plt.xlabel('Feature Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def process_data(file_path=None, plot=False):
    inicio = time.time()
    df = import_data(file_path)
    df = preprocess_data(df)
    features = rfe_selection(df.drop(columns=df.columns[-1]), df[df.columns[-1]], n_features_to_select=10)
    df = df[features + [df.columns[-1]]]  #Keep only selected features
    fim = time.time()
    print(f"Data processing completed in {fim - inicio:.2f} seconds.")
    xtrain, xtest, ytrain, ytest = split_data(df, target_column=df.columns[-1])
    if plot:
        plot_data_distribution(df, target_column=df.columns[-1])
        plot_features_selection(df, features)
        plot_data_selection(xtrain, ytrain)
    return (xtrain, xtest, ytrain, ytest)

if __name__ == "__main__":
    print("Starting data preparation...")
    xtrain, xtest, ytrain, ytest = process_data(plot=True)
    print("Training and testing data prepared successfully.")
    plot_data_selection(xtrain, ytrain)
    plot_data_selection(xtest, ytest)
    print(f"Training set size: {xtrain.shape[0]}, Testing set size: {xtest.shape[0]}")
    print(f"Selected features: {xtrain.columns.tolist()}")