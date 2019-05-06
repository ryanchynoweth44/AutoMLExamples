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


MLBox is similar to other Automated Machine Learning libraries since it does not automate the data science process. MLBox simply helps developers create the optimal model and select the best features to make predictions for the label of your choice. 

Lets get started using the MLBox library! 

### Developing Using MLBox

#### Installing MLBox
For this demo we will be using Anaconda Virtual Environments, and I will be using Visual Studio Code as my IDE. For more information on how to use the Anaconda distribution with Visual Studio code check out this [blog](https://ryansdataspot.com/2019/02/14/anaconda-environments-in-visual-studio-code/) I wrote. Additionally, I will be developing on a windows machine which is currently in an experimental release.  

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


