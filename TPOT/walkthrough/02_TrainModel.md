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

1. Since we already have a test file, we will split our training dataset into train and validation. Additionally, since TPOT is built on scikit-learn our input data sources need to be numpy arrays, therefore, we convert them using the `.values`.
    ```python
    # split data to train and validate
    train, validate = train_test_split(data, test_size=.25)
    X_train = train.drop('class', axis=1).values
    y_train = train['class'].values
    X_validate = validate.drop('class', axis=1).values
    y_validate = validate['class'].values
    ```

1. To fit our model we simply identify the type of model we want to train, then fit the model as you normally would. One great parameter option is the `max_time_mins` parameter because Auto ML libraries can take a very long time to train the best model. The `max_time_mins` will stop the model training after a specified amount of time and return the best model that it has trained thus far. 
    ```python
    # fit our model
    # max_time_mins will stop the auto ml early
    tpot = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=40)
    tpot.fit(X_train, y_train)
    ```

1. We can know take our trained model and compare it to our validation set. 
    ```python
    validate = tpot.score(X_validate, y_validate)
    ```

1. The similarly to other python libraries we can serialize our model and save it to file so that we can use it later.  
    ```python
    # export the model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=tpot.fitted_pipeline_, filename='outputs/best_model.pkl')
    ```

1. To test and make sure that we can actually load the model we saved, we will reload it and predict on the validate set once more.  
    ```python
    # load the best model and predict
    model = joblib.load('outputs/best_model.pkl')
    model.predict(X_validate)
    ```

1. One of the best features about TPOT is that you can export the best model as a python training script. This gives you the option to not have to use Auto ML to retrain your model. 
    ```python
    tpot.export('tpot_train_model.py')
    ```


