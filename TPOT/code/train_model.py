# Import required libraries
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pandas as pd 
import numpy as np
import os

# read data
data = pd.read_csv('data/titanic_train.csv')

# rename our label column to 'class' for tpot
data.rename(columns={'Survived': 'class'}, inplace=True)

# encode some of our categorical variables
data['Sex'] = data['Sex'].map({'male':0,'female':1})
data['Embarked'] = data['Embarked'].map({'S':0,'C':1,'Q':2})


# Replace null with -999 as a placeholder
data = data.fillna(-999)
# check if we missed any nulls 
pd.isnull(data).any()

# drop the cols we don't really care about
data = data.drop(['Name','Ticket','Cabin'], axis=1)

# split data to train and validate
train, validate = train_test_split(data, test_size=.25)
X_train = train.drop('class', axis=1).values
y_train = train['class'].values
X_validate = validate.drop('class', axis=1).values
y_validate = validate['class'].values


# fit our model
# max_time_mins will stop the auto ml early
tpot = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=40)
tpot.fit(X_train, y_train)

# compare against our validate
validate = tpot.score(X_validate, y_validate)

# export the model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=tpot.fitted_pipeline_, filename='outputs/best_model.pkl')

# loading and using the best model
model = joblib.load('outputs/best_model.pkl')
model.predict(X_validate)

# export tpot training script
tpot.export('tpot_train_model.py')