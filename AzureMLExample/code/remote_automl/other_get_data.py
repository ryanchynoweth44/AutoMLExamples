from azure.storage.blob import BlockBlobService
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
from io import StringIO

from app_helper import AppHelper
helper = AppHelper()


def get_data():
    ## Read data from blob and load into pandas data frame
    blob_account = BlockBlobService(account_name=helper.storage_name, account_key=helper.storage_key)

    bytes_data = blob_account.get_blob_to_bytes(container_name=helper.azure_data_container, blob_name=helper.azure_data_path)

    data = StringIO(bytes_data.content.decode('utf-8')) 

    df = pd.read_csv(data)

    train, test = train_test_split(df, test_size=.25)
    X_train = train.drop('tip_amount', axis=1).values
    Y_train = train['tip_amount'].values
    return { "X" : X_train, "Y" : Y_train }
