import pandas as pd
import numpy as np
from azure.storage.blob import BlockBlobService
import azureml.core
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.train.automl import AutoMLConfig
import logging, os
from sklearn.model_selection import train_test_split

from app_helper import AppHelper
helper = AppHelper()


## Connect to our Azure Machine Learning Workspace
auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )

## Experiment name and project folder
experiment_name = 'auto-ml-demo'
project_folder = './auto-ml-demo'

exp = Experiment(ws, experiment_name)

## Read data from blob and load into pandas data frame
blob_account = BlockBlobService(account_name=helper.storage_name, account_key=helper.storage_key)

if os.path.exists(helper.local_data_path) == False :
    print("Downloading training data.")
    os.makedirs('./data', exist_ok=True)
    blob_account.get_blob_to_path(container_name=helper.azure_data_container, blob_name=helper.azure_data_path, file_path=helper.local_data_path)

df = pd.read_csv(helper.local_data_path)


automl_settings = {
    "iteration_timeout_minutes" : 20,
    "iterations" : 5,
    "primary_metric" : 'r2_score',
    "preprocess" : True,
    "verbosity" : logging.INFO,
    "n_cross_validations": 5
}

train, test = train_test_split(df, test_size=.25)
X_train = train.drop('tip_amount', axis=1).values
Y_train = train['tip_amount'].values.flatten()
X_test = test.drop('tip_amount', axis=1).values
Y_test = test['tip_amount'].values.flatten()


# local compute
## note here that the input x,y datasets are not Pandas!
## these are numpy arrays therefore you will have to do 
## prework in pandas prior to sending it to auto ml
automated_ml_config = AutoMLConfig(task = 'regression',
                             debug_log = 'automated_ml_errors.log',
                             path = project_folder,
                             X = X_train,
                             y = Y_train,
                             **automl_settings)



local_run = exp.submit(automated_ml_config, tags={'Category': 'AutoMLExample'}, show_output=True)

### convert the runs to a pandas dataframe  
runs = list(local_run.get_children())
all_metrics = {}

for run in runs:
    properties = run.get_properties()

    metrics = {k: v for k, v  in run.get_metrics().items() if isinstance(v, float)}
    all_metrics[int(properties['iteration'])] = metrics

run_data = pd.DataFrame(all_metrics)


best_run, best_model = local_run.get_output()


os.makedirs('outputs', exist_ok=True)

# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=best_model, filename='outputs/auto_ml_model.pkl')


# upload the model file explicitly into artifacts 
local_run.upload_file(name = 'auto_ml_model.pkl', path_or_stream = 'outputs/auto_ml_model.pkl')
# register the model 
local_run.register_model(model_name = 'auto_ml_model.pkl', model_path = 'outputs/auto_ml_model.pkl' )

