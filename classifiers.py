import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
import plot as plt
import time

##########################################
def random_forest(X, y):
    """
    Train a Random Forest classifier.
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    Returns:
    clf (RandomForestClassifier): Trained Random Forest classifier.
    """
    print("\n---Train Random Forest Classifier---")
    start = time.time()
    clf = sklearn.ensemble.RandomForestClassifier()
    clf.fit(X, y)
    end = time.time()
    print(f"Random Forest training completed in {end - start:.2f} seconds.\n")
    return clf

##########################################
def knn(X, y):
    """
    Train a K-Nearest Neighbors classifier.
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    Returns:
    clf (KNeighborsClassifier): Trained K-Nearest Neighbors classifier.
    """
    print("\n---Train K-Nearest Neighbors Classifier---")
    start = time.time()
    clf = sklearn.neighbors.KNeighborsClassifier()
    clf.fit(X, y)
    end = time.time()
    print(f"KNN training completed in {end - start:.2f} seconds.\n")
    return clf 

##########################################
def neural_network(X, y, hidden_layer_sizes=(15,15), max_iter=1500, random_state=42):
    """
    Train a Multi-layer Perceptron classifier.
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    hidden_layer_sizes (tuple): The ith element represents the number of neurons in the ith hidden layer.
    max_iter (int): Maximum number of iterations.
    Returns:
    clf (MLPClassifier): Trained Multi-layer Perceptron classifier.
    """
    print("\n---Train Multi-layer Perceptron Classifier---")
    inicio = time.time()
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
    clf.fit(X, y)
    fim = time.time()
    print(f"Neural network training completed in {fim - inicio:.2f} seconds.\n")
    return clf

##########################################
def evaluate_classifier(clf, X_test, y_test, name='Classifier', plot=False):
    
    """
    Evaluate the classifier on the test set.
    Parameters:
    clf (classifier): Trained classifier.
    X_test (array-like): Test feature matrix.
    y_test (array-like): True labels for the test set.
    Shows confusion matrix, classification report, accuracy, F1 score, precision, and recall.
    """
    print(f"\n---Evaluate {name}---")
    y_pred = clf.predict(X_test)
    print("\nConfusion Matrix:\n", sklearn.metrics.confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", sklearn.metrics.classification_report(y_test, y_pred))
    print("\nAccuracy Score:", sklearn.metrics.accuracy_score(y_test, y_pred))
    print("\nF1 Score:", sklearn.metrics.f1_score(y_test, y_pred, average='weighted'))
    print("\nPrecision Score:", sklearn.metrics.precision_score(y_test, y_pred, average='weighted'))
    print("\nRecall Score:", sklearn.metrics.recall_score(y_test, y_pred, average='weighted'))

    if plot:
        if X_test.shape[1] >= 2:
            X_test_2d = X_test.iloc[:, :2]
            temp_clf = clone(clf) # Create a new instance of the same type of classifier
            temp_clf.fit(X_test_2d, y_test) # Train it on the 2D data
            
            plt.plot_decision_boundary(temp_clf, X_test_2d, y_test, title=f'{name} Decision Boundary')
        else:
            print("Skipping decision boundary plot: Not enough features (less than 2) in X_test.")
            
        #plt.plot_classifier_results(y_test, y_pred, title=f'{name} Results')