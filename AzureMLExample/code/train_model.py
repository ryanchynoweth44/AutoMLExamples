import pandas as pd
import numpy as np
from azure.storage.blob import BlockBlobService
import azureml.core
from azureml.core.workspace import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
import logging, os
from sklearn.model_selection import train_test_split

from app_helper import AppHelper
helper = AppHelper()

## Read data from blob and load into pandas data frame
blob_account = BlockBlobService(account_name=helper.storage_name, account_key=helper.storage_key)

if os.path.exists(helper.local_data_path) == False :
    os.makedirs('./data', exist_ok=True)
    blob_account.get_blob_to_path(container_name=helper.azure_data_container, blob_name=helper.azure_data_path, file_path=helper.local_data_path)

df = pd.read_csv(helper.local_data_path)

## Connect to our Azure Machine Learning Workspace
auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )

## Experiment name and project folder
experiment_name = 'auto-ml-demo'
project_folder = './auto-ml-demo'







### https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-auto-train-models#start

### https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python