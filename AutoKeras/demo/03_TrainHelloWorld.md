# Train a model with the MNIST Dataset

The most common deep learning example dataset available is [MNIST](http://yann.lecun.com/exdb/mnist/). It is a dataset composed of 60,000 handwritten numbers and is widely used as an example deep learning solution that predicts the digit in the image. MNIST is also the most common dataset example used to show off the capabilities of AutoKeras, so lets quickly go over it as an example.  

1. Import the required libraries for this example.  
    ```python
    from keras.datasets import mnist
    from autokeras import ImageClassifier
    import os 
    ```

1. Load the MNIST dataset from keras.  
    ```python
    # loading mnist from keras
    (X_train, y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape + (1,))
    X_test = X_test.reshape(X_test.shape + (1,))
    ```

1. Lets quickly initialize our classifier. I would recommend always setting `verbose=True`, otherwise, you won't get much output while training a model and training a neural network can take a bit of time.  
    ```python
    clf = ImageClassifier(verbose=True)
    ```

1. Let's `fit` out model and perform a `final_fit` as well. For now we will use a 10 minute time limit since this is a quick example.    
    ```python
    # fit the classifier to the dataset
    clf.fit(X_train, Y_train, time_limit=600)
    clf.final_fit(X_train, Y_train, X_test, Y_test, retrain=True)
    ```

1. Now we will evaluate using our test dataset.  
    ```python
    Y_pred = clf.evaluate(X_test, Y_test)
    print(Y_pred)

    os.makedirs('mnist_model', exist_ok=True)
    clf.export_autokeras_model('mnist_model/autokeras_mnist_model.h5')
    clf.export_keras_model('mnist_model/keras_model_mnist_model.h5')
    clf.load_searcher().load_best_model().produce_keras_model().save('mnist_model/best_keras_model.h5')
    ```


