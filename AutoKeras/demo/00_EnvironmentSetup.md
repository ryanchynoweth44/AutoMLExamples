# Setting up Anaconda Environment

Please complete the following instructions to setup an Anaconda Environment for AutoKeras development. Please note that the [Anaconda](https://anaconda.com) Python distribution must be installed in order to complete the steps below.    

1. Create a conda environment running Python 3.6.
    ```
    conda create -n autokerasenv python=3.6

    conda activate autokerasenv
    ```


1. To develop using AutoKeras I would recommend using a Linux machine so that you can easily run `pip install autokeras` to get started. 

1. I would also recommend install the `keras` and `tensorflow` libraries as well so that we can utilize additional functions, and we will be training a model using the `keras` prior to training with `autokeras`. 
    ```
    pip install keras

    pip install tensorflow
    ## gpu version of tensorflow
    # pip install tensorflow-gpu

    pip install pillow

    pip install pandas    
    ```


You are ready to get started training a neural network model with AutoKeras and Keras! Move onto the next portion of the demo to [download a training dataset](./01_DownloadImages.md).  