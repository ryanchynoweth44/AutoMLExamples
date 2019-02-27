from azure.storage.blob import BlockBlobService
import os
import pandas as pd
import numpy as np
from io import StringIO

from app_helper import AppHelper
helper = AppHelper()


def get_data():
    ## Read data from blob and load into pandas data frame
    # print("---------------- Connecting to blob")
    # blob_account = BlockBlobService(account_name=helper.storage_name, account_key=helper.storage_key)
    
    # print("---------------- reading blob to bytes")
    # bytes_data = blob_account.get_blob_to_bytes(container_name=helper.azure_data_container, blob_name=helper.azure_data_path)

    # print("---------------- Connecting to blob")
    # data = StringIO(bytes_data.content.decode('utf-8')) 

    print("---------------- Reading data")
    df = pd.read_csv("https://rserverdata.blob.core.windows.net/public/nyctaxitip.csv")

    X_df = df.drop(['tip_amount'], axis=1)
    Y_df = df['tip_amount'].values
    return { "X" : X_df, "y" : Y_df }

### https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-vscode-tools