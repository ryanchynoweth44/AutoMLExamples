## Training a Machine Learning Model with TPOT

In this section of the walk through we will be automatically be training a model locally to predict the titanic survival to compare it to the previously posted Auto ML with Azure ML blog post.  

The dataset that we are using is the popular titanic dataset where we use information about each passenger to predict whether or not they survived the catastrophe. For more information about the dataset check out the [Kaggle Competition](https://www.kaggle.com/c/titanic).  

1. We will be using the Titanic Dataset. Please download the data from the GitHub repository [here](https://github.com/ryanchynoweth44/AutoMLExamples/tree/master/data). Save the file to a `data` folder in your application directory. Please note that the application directory I will be using is the `TPOT` directory in my repository. 

1. Next we will import all the libraries we need to develop our machine learning model.  
    ```python
    # Import required libraries
    from tpot import TPOTClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.externals import joblib
    import pandas as pd 
    import numpy as np
    import os
    ```

1. We will read our CSV file into a pandas dataframe. 
    ```python
    # read data
    data = pd.read_csv('data/titanic_train.csv')
    ```

1. TPOT requires our label column to be called `class` when developing classification models, therefore, we need to rename the column. 
    ```python
    # rename our label column to 'class' for tpot
    data.rename(columns={'Survived': 'class'}, inplace=True)
    ```

1. We have a few categorical variables that we will want to convert from raw text to an encoding so that we can use them in our training process. We will simply do this manually using a map function.  
    ```python
    # encode some of our categorical variables
    data['Sex'] = data['Sex'].map({'male':0,'female':1})
    data['Embarked'] = data['Embarked'].map({'S':0,'C':1,'Q':2})
    ```

1. We will replace any null values with -999, then drop the columns we don't need. 
    ```python
    # fill nulls
    data = data.fillna(-999)
    # drop cols
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    ```


