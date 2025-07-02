import data_preparation as dp 
import classifiers as cs

if __name__ == "__main__":

  xtrain, xtest, ytrain, ytest = dp.process_data() 

  knn=cs.knn(xtrain, ytrain)
  cs.plot_decision_boundary(xtrain, ytrain, knn, title='KNN Decision Boundary')
  print("KNN Accuracy:", cs.evaluate_classifier(knn, xtest, ytest))
  rf=cs.random_forest(xtrain, ytrain)
  cs.plot_decision_boundary(xtrain, ytrain, rf, title='Random Forest Decision Boundary')
  print("Random Forest Accuracy:", cs.evaluate_classifier(rf, xtest, ytest))
  mlp=cs.neural_network(xtrain, ytrain)
  cs.plot_decision_boundary(xtrain, ytrain, mlp, title='MLP Decision Boundary')
  print("MLP Accuracy:", cs.evaluate_classifier(mlp, xtest, ytest))