import data_preparation as dp 
import classifiers as cs

if __name__ == "__main__":

  """First Execution - Import Data:"""
  #xtrain, xtest, ytrain, ytest = dp.process_data(file_path = 'Blind_data_classifier/bases', plot=True, save=True, first_execution=True, n_features_to_select=100)

  """Second Execution:"""
  df=dp.import_data('Blind_data_classifier/processed_data.csv') 
  xtrain, xtest, ytrain, ytest=dp.split_data(df, target_column=df.columns[-1])
  #dp.plot_train_test_data(xtrain, ytrain, xtest, ytest, title='Train and Test Data Distribution')

  """Classifiers:"""
  
  knn=cs.knn(xtrain, ytrain)
  cs.evaluate_classifier(knn, xtest, ytest, name='KNN Classifier',plot=True)

  rf=cs.random_forest(xtrain, ytrain)
  cs.evaluate_classifier(rf, xtest, ytest, name='Random Forest Classifier', plot=True)

  mlp=cs.neural_network(xtrain, ytrain)
  cs.evaluate_classifier(mlp, xtest, ytest, name='MLP Classifier', plot=True)
