from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression, ClassificationTree
from interpret import show
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os, sys



train_data = pd.read_csv('../data/titanic_train.csv')
test_data = pd.read_csv('../data/titanic_test.csv')

train_data = train_data.fillna(train_data.groupby(['Pclass', 'Sex']).transform('mean'))
test_data = test_data.fillna(test_data.groupby(['Pclass', 'Sex']).transform('mean'))

train_data = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Survived']]
test_data = test_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]

X_train, X_validate, y_train, y_validate = train_test_split(train_data.drop('Survived', axis=1), train_data['Survived'], test_size = .25)


ebm = ExplainableBoostingClassifier()
lrm = LogisticRegression()

ebm.fit(X_train, y_train)

le = LabelEncoder()
X_train_lr = X_train
X_train_lr['Sex'] = le.fit_transform(X_train['Sex'])
lrm.fit(X_train_lr, y_train)


ebm_global = ebm.explain_global()
show(ebm_global)
ebm_local = ebm.explain_local(X_validate, y_validate)
show(ebm_local)

lrm_global = lrm.explain_global()
show(lrm_global)
X_validate_lr = X_validate
X_validate_lr['Sex'] = le.fit_transform(X_validate['Sex'])
lrm_local = lrm.explain_local(X_validate, y_validate)
show(lrm_local)


## Age binning
ages = pd.DataFrame({'ages': [10, 20, 24, 25, 29, 41, 45, 55, 56]})
ages['ages2'] = pd.cut(ages.ages, bins=[0,20,40,60], include_lowest=True)
ages
