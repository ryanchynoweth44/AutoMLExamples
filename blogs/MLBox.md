## Automated Machine Learning with MLBox

In continuation of my [AutoML Blog Series](https://ryansdataspot.com/2019/03/01/automated-machine-learning/) we will be evaluating the capabilities of MLBox

### What is MLBox

MLBox is an extremely popular and powerful automated machine learning python library. As noted by the [MLBox Documentation](https://mlbox.readthedocs.io/en/latest/index.html), it provides features for:

- Fast reading of data
- Distributed data processing
- Robust feature selection
- Accurate hyper-parameter tuning
- Start-of-the are machine learning and deep learning models
- Model interpretation 


MLBox is similar to other Automated Machine Learning libraries as it does not automate the data science process, but augments a developers ability to quickly create machine learning models. MLBox simply helps developers create the optimal model and select the best features to make predictions for the label of your choice. 

One draw back of the MLBox library is that it doesn't necessarily conform to a data scientist's process, rather, the data scientist has to work the way the library expects. One example, is that I will often use three datasets when developing machine learning solutions in an attempt to avoid overfitting: train, validation, and test. Having these three datasets is rather difficult to do with MLBox.    

Lets get started using the MLBox library! 

### Developing with MLBox

#### Installing MLBox
For this demo we will be using Anaconda Virtual Environments, and I will be using Visual Studio Code as my IDE. For more information on how to use the Anaconda distribution with Visual Studio code check out this [blog](https://ryansdataspot.com/2019/02/14/anaconda-environments-in-visual-studio-code/) I wrote. Additionally, I will be developing on a windows machine which is currently in an experimental release.  

We will also need a linux machine to do our MLBox development, if you do not have one available you can create one by following these [instructions](https://ryansdataspot.com/2019/05/07/linux-development-from-a-windows-guy/). 

1. First let's create a new Anaconda Environment.
    ```
    conda create -n MLBoxEnv python=3.6

    conda activate MLBoxEnv
    ```

1. Next we will run the following installs.
    ```
    pip install setuptools

    pip install mlbox
    ```

#### Training a Model
As with the other AutoML libraries we will be using the titanic dataset where we will use specific features to predict whether or not they survived the catastrophe. For more information about the dataset check out the [Kaggle Competition](https://www.kaggle.com/c/titanic).  

1. Please download the data from the GitHub repository [here](https://github.com/ryanchynoweth44/AutoMLExamples/tree/master/data). Save the file to a `data` folder in your application directory. Please note that the application directory I will be using is the `TPOT` directory in my repository. 


1. Now that we have our data and MLBox installed, let's read our datasets into memory and start preparing it for a machine learning algorithm. 

    MLBox has its own Reader class for efficient and distributed reading of data, one key feature to this class is that it expects a list of file paths to your training and test datasets. Interacting with my datasets was slightly foreign at first, but once I learned that the Reader class creates a dictionary object with pandas dataframes and our target (label) column as a pandas series it was easier to work with. 

    ```python
    from mlbox.preprocessing import *
    from mlbox.optimisation import *
    from mlbox.prediction import *


    train_path = ["./data/titanic_train.csv", "./data/titanic_test.csv"]

    reader = Reader(sep=",", header=0)
    data = reader.train_test_split(train_path, 'Survived')
    ```

    There are a few things worth noting about the `train_test_split` function. The dataset is only considered to be a test set if there is no label column present, otherwise, it will be merged with a train set. Being able to provide a list of file paths is a nice feature to have because it can allow developers to easily ingest many files at once, which is common with bigger datasets and data lakes. Since the function automatically scans for the target column there is little work for the developer to even identify a test dataset. Additionally, it determines whether or not it is a regression or classification problem based off our label and will automatically encode the column as needed. 

1. One really nice feature of MLBox is the ability to automatically remove drift variables. I am not an expert when it comes to explaining what drift is by all means, however, drift is the idea that the process or observation behavior may change over time. In turn the data will slowly change resulting in the relationship between the features to change as well. MLBox has built in functionality to deal with this drift. We will use a drift transform.  
    ```python
    data = Drift_thresholder().fit_transform(data)
    ```


1. As with all Automated Machine Learning libraries, the key feature is not necessarily the algorithms but is the data scientist's ability to select the appropriate features and optimal hyper-parameters for the algorithm. Using MLBox's `Optimiser` class we are able to create a dimensional space to figure out the best set of parameters. Therefore, to optimize  we must create a parameter space, and select the scoring metric we wish to optimize. 
    ```python
    opt = Optimiser(scoring='accuracy', n_folds=3)
    opt.evaluate(None, data)

    space = {
        'ne__numerical_strategy': {"search":"choice", "space":[0]},
        'ce__strategy': {"search":"choice", "space":["label_encoding", "random_projection", "entity_embedding"]},
        'fs__threshold': {"search":"uniform", "space":[0.01, 0.3]},
        'est__max_depth': {"search":"choice", "space":[3,4,5,6,7]}
    }

    best_params = opt.optimise(space, data, 10)
    ```

1. Next we can use the Predictor class to train a machine learning model. 
    ```
    model = Predictor().fit_predict(best_params, data)
    ```
    The line of code above will create a folder called `save` so that it can export an sklearn pipeline that you can reuse for model deployment or further validation. Additionally, it provides you exports of feature importance, a csv of test predictions, and a target encoder object so that you can map the encoded values back to their original values.  



For more information on MLBox, please check out their [Github](https://github.com/AxeldeRomblay/MLBox) repository or the [official documentation](https://mlbox.readthedocs.io/en/latest/) page. MLBox is a great library to assist data scientists in building a machine learning solution. For a full copy of the demo python file please refer to my personal [Github](https://github.com/ryanchynoweth44/AutoMLExamples/blob/master/MLBox/TrainModel.py). 