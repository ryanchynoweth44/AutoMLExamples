from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np



def get_data():
    ## Read data from blob and load into pandas data frame
    df = pd.read_csv('data/nyctaxitip.csv')

    train, test = train_test_split(df, test_size=.25)
    X_train = train.drop('tip_amount', axis=1).values
    Y_train = train['tip_amount'].values
    return { "X" : X_train, "Y" : Y_train }
