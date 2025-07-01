import sklearn
from sklearn.neural_network import MLPClassifier

def random_forest(X, y):
    """
    Train a Random Forest classifier.
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    Returns:
    clf (RandomForestClassifier): Trained Random Forest classifier.
    """
    clf = sklearn.ensemble.RandomForestClassifier()
    clf.fit(X, y)
    return clf

def knn(X, y):
    """
    Train a K-Nearest Neighbors classifier.
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    Returns:
    clf (KNeighborsClassifier): Trained K-Nearest Neighbors classifier.
    """
    clf = sklearn.neighbors.KNeighborsClassifier()
    clf.fit(X, y)
    return clf 

def neural_network(X, y, hidden_layer_sizes=(100,), max_iter=200):
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
    clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter)
    clf.fit(X, y)
    return clf

def evaluate_classifier(clf, X_test, y_test):
    """
    Evaluate the classifier on the test set.
    Parameters:
    clf (classifier): Trained classifier.
    X_test (array-like): Test feature matrix.
    y_test (array-like): True labels for the test set.
    Returns:
    accuracy (float): Accuracy of the classifier on the test set.
    """
    y_pred = clf.predict(X_test)
    return sklearn.metrics.accuracy_score(y_test, y_pred)