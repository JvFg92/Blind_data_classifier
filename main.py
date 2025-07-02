import data_preparation as dp 
import classifiers as cs

if __name__ == "__main__":

  xtrain, xtest, ytrain, ytest = dp.process_data() 

  knn=cs.knn(xtrain, ytrain)
  print("KNN Accuracy:", cs.evaluate_classifier(knn, xtest, ytest))
  rf=cs.random_forest(xtrain, ytrain)
  print("Random Forest Accuracy:", cs.evaluate_classifier(rf, xtest, ytest))
  mlp=cs.neural_network(xtrain, ytrain)
  print("MLP Accuracy:", cs.evaluate_classifier(mlp, xtest, ytest))