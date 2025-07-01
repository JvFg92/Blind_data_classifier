import data_preparation as dp
import tensorflow as tf

class mlp_classifier:
  def __init__(self, input_shape, num_classes, file_path=None):
      """
      Initialize the MLP classifier.
      
      Parameters:
      input_shape (tuple): Shape of the input data.
      num_classes (int): Number of output classes.
      """
      if file_path:
          # Import and preprocess data if file_path is provided
          df = dp.import_data(file_path)
          df = dp.preprocess_data(df)
          X_train, X_test, y_train, y_test = dp.split_data(df, target_column='target')
          
          self.X_train = X_train.values
          self.y_train = y_train.values
          self.X_test = X_test.values
          self.y_test = y_test.values

      self.model = tf.keras.Sequential([
          tf.keras.layers.Input(shape=input_shape),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dropout(0.2),
          tf.keras.layers.Dense(num_classes, activation='softmax')
      ])
      
      self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
      self.model.summary()

  def train(self, epochs=10, batch_size=32, validation_split=0.2):
     self.history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)


