import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone
import plot as plt
import time
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, f1_score


##########################################
def train_and_tune_classifier(estimator, param_grid, X, y, scoring_metric='f1_weighted', cv_folds=5):
    """
    Trains and tunes a classifier using GridSearchCV.

    Parameters:
    estimator (estimator): The classifier estimator (e.g., RandomForestClassifier(), KNeighborsClassifier()).
    param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings to try as values.
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    scoring_metric (str): The scoring strategy to use for evaluating the models. Defaults to 'f1_weighted'.
                          'f1_weighted' is chosen because it accounts for class imbalance by calculating the F1 score
                          for each label and then averaging, weighted by support (the number of true instances for each label).
                          This provides a balanced measure that considers both precision and recall across all classes.
    cv_folds (int): Number of folds in a (Stratified)KFold cross-validation.

    Returns:
    best_estimator (estimator): The best estimator found by GridSearchCV.
    best_params (dict): Dictionary of the best parameters found.
    """
    print(f"\n--- Starting GridSearchCV for {estimator.__class__.__name__} ---")
    start = time.time()

    # Use StratifiedKFold for classification tasks to ensure each fold has a similar class distribution
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # If the scoring metric is f1_weighted, use make_scorer to ensure proper handling for multi-class
    scorer = make_scorer(f1_score, average='weighted') if scoring_metric == 'f1_weighted' else scoring_metric

    grid_search = GridSearchCV(estimator, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(X, y)

    end = time.time()
    print(f"GridSearchCV for {estimator.__class__.__name__} completed in {end - start:.2f} seconds.")
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best {scoring_metric} score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_

##########################################
def random_forest(X, y):
    """
    Trains a Random Forest classifier with hyperparameter tuning.
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    Returns:
    clf (RandomForestClassifier): Trained Random Forest classifier with best hyperparameters.
    """
    print("\n---Train Random Forest Classifier---")
    # Define the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_features': ['sqrt', 'log2'],  # Number of features to consider when looking for the best split
        'max_depth': [10, 20, None],  # Maximum depth of the tree
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
    }
    rf_estimator = RandomForestClassifier(random_state=42)
    best_rf, _ = train_and_tune_classifier(rf_estimator, param_grid, X, y)
    return best_rf

##########################################
def knn(X, y):
    """
    Trains a K-Nearest Neighbors classifier with hyperparameter tuning.
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    Returns:
    clf (KNeighborsClassifier): Trained K-Nearest Neighbors classifier with best hyperparameters.
    """
    print("\n---Train K-Nearest Neighbors Classifier---")
    # Define the parameter grid for KNN
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],  # Number of neighbors to consider
        'weights': ['uniform', 'distance'],  # Weight function used in prediction
        'p': [1, 2]  # Power parameter for the Minkowski metric (1 for Manhattan, 2 for Euclidean)
    }
    knn_estimator = KNeighborsClassifier()
    best_knn, _ = train_and_tune_classifier(knn_estimator, param_grid, X, y)
    return best_knn

##########################################
def neural_network(X, y):
    """
    Trains a Multi-layer Perceptron classifier with hyperparameter tuning.
    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.
    Returns:
    clf (MLPClassifier): Trained Multi-layer Perceptron classifier with best hyperparameters.
    """
    print("\n---Train Multi-layer Perceptron Classifier---")
    # Define the parameter grid for MLPClassifier
    param_grid = {
        'hidden_layer_sizes': [(15,14)],  # Number of neurons in each hidden layer
        'activation': ['tanh', 'relu'],  # Activation function for the hidden layer
        'solver': ['adam', 'sgd'],  # The solver for weight optimization
        'alpha': [0.0001, 0.001, 0.01],  # L2 penalty (regularization term) parameter
        'learning_rate': ['constant', 'adaptive'] # Learning rate schedule
    }
    mlp_estimator = MLPClassifier(max_iter=1500, random_state=42) # Increased max_iter for convergence
    best_mlp, _ = train_and_tune_classifier(mlp_estimator, param_grid, X, y)
    return best_mlp

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
            # For plotting decision boundary, we might need to train a temporary classifier
            # on a 2D subset of the data, as decision boundary plots are typically 2D.
            # This is a simplification and might not perfectly represent the high-dimensional
            # decision boundary, but gives a visual intuition.
            X_test_2d = X_test.iloc[:, :2]
            
            # Create a new instance of the same type of classifier with the best parameters
            # and train it on the 2D data for visualization purposes.
            temp_clf = clone(clf) 
            
            # If the classifier has a 'set_params' method (which most sklearn estimators do),
            # we can set the best parameters found during grid search for this temp_clf.
            # This makes the 2D plot more representative of the tuned model.
            try:
                # Get the parameters that were actually used by the best_estimator
                params_for_2d_plot = clf.get_params()
                temp_clf.set_params(**params_for_2d_plot)
            except AttributeError:
                print("Warning: Classifier does not have 'get_params' or 'set_params' method. Decision boundary plot might not reflect tuned parameters fully.")

            temp_clf.fit(X_test_2d, y_test)
            
            plt.plot_decision_boundary(temp_clf, X_test_2d, y_test, title=f'{name} Decision Boundary')
        else:
            print("Skipping decision boundary plot: Not enough features (less than 2) in X_test.")
            
        #plt.plot_classifier_results(y_test, y_pred, title=f'{name} Results')