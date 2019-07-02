from keras.datasets import mnist
from autokeras import ImageClassifier
import os 

# loading mnist from keras
(X_train, y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape + (1,))
X_test = X_test.reshape(X_test.shape + (1,))

# initialize the classifier
clf = ImageClassifier(verbose=True)

# fit the classifier to the dataset
clf.fit(X_train, Y_train, time_limit=10)
clf.final_fit(X_train, Y_train, X_test, Y_test, retrain=True)



Y_pred = clf.evaluate(X_test, Y_test)
print(Y_pred)

os.makedirs('mnist_model', exist_ok=True)
clf.export_autokeras_model('mnist_model/autokeras_mnist_model.h5')
clf.export_keras_model('mnist_model/keras_model_mnist_model.h5')
clf.load_searcher().load_best_model().produce_keras_model().save('mnist_model/best_keras_model.h5')

