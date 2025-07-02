import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def plot_train_test_data(Xtrain, Ytrain, Xtest, Ytest, title='Dados de Treino e Teste'):
    """
    Plota os dados de treino e teste com cores distintas para cada conjunto.
    
    Parameters:
    Xtrain (DataFrame): Matriz de features de treino.
    Ytrain (Series): Variável alvo de treino.
    Xtest (DataFrame): Matriz de features de teste.
    Ytest (Series): Variável alvo de teste.
    """
    plt.figure(figsize=(10, 6))
    
    # Atribui cores fixas para treino e teste
    plt.scatter(Xtrain.iloc[:, 0], Xtrain.iloc[:, 1], color='blue', label='Treino', alpha=0.5)
    plt.scatter(Xtest.iloc[:, 0], Xtest.iloc[:, 1], color='red', label='Teste', alpha=0.5)
    
    plt.title(title)
    plt.xlabel(Xtrain.columns[0])
    plt.ylabel(Xtrain.columns[1])
    plt.legend()
    plt.grid(True) # Adiciona um grid para melhor visualização
    plt.show()

def plot_classifier_results(y_true, y_pred, title='Classifier Results'):
    """
    Plot the results of the classifier predictions.
    
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.title(title)
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Diagonal line
    plt.grid(True)
    plt.show()
  
def plot_decision_boundary(clf, X, y, title='Decision Boundary'):
    """
    Plot the decision boundary of a classifier.
    
    Parameters:
    clf (classifier): Trained classifier.
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    title (str): Title of the plot.
    """
    # Convert X to a NumPy array if it's a Pandas DataFrame
    if isinstance(X, pd.DataFrame): # Import pandas as pd at the top of plot.py if not already
        X = X.values
    if isinstance(y, pd.Series): # Import pandas as pd at the top of plot.py if not already
        y = y.values

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()