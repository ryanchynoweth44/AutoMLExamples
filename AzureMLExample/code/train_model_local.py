import pandas as pd
import numpy as np
from azure.storage.blob import BlockBlobService
import azureml.core
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.train.automl import AutoMLConfig
import azureml.dataprep as dprep
import logging, os
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

from app_helper import AppHelper
helper = AppHelper()

## Connect to our Azure Machine Learning Workspace
auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )

## Experiment name and project folder
experiment_name = 'local-auto-ml-demo'
project_folder = './local-auto-ml-demo'

exp = Experiment(ws, experiment_name)

## Read data from blob and load into pandas data frame
blob_account = BlockBlobService(account_name=helper.storage_name, account_key=helper.storage_key)

if os.path.exists(helper.local_data_path) == False :
    print("Downloading training data.")
    os.makedirs('./data', exist_ok=True)
    blob_account.get_blob_to_path(container_name=helper.azure_data_container, blob_name=helper.azure_data_path, file_path=helper.local_data_path)


df = pd.read_csv(helper.local_data_path)
train, test = train_test_split(df, test_size=.25)
X_train = train.drop('Survived', axis=1).values
Y_train = train['Survived'].values
X_test = test.drop('Survived', axis=1).values
Y_test = test['Survived'].values


automl_settings = {
    "iteration_timeout_minutes" : 60,
    "iterations" : 5,
    "primary_metric" : 'accuracy',
    "preprocess" : True,
    "verbosity" : logging.INFO,
    "n_cross_validations": 5
}

# local compute
## note here that the input x,y datasets are not Pandas!
## these are numpy arrays therefore you will have to do 
## prework in pandas prior to sending it to auto ml
automated_ml_config = AutoMLConfig(task = 'classification',
                             debug_log = 'automated_ml_errors.log',
                             path = project_folder,
                             X = X_train,
                             y = Y_train,
                             **automl_settings)



local_run = exp.submit(automated_ml_config, tags={'Category': 'AutoMLExample'}, show_output=True)



#################################################################
### convert the runs to a pandas dataframe  
runs = list(local_run.get_children())
all_metrics = {}

for run in runs:
    properties = run.get_properties()

    metrics = {k: v for k, v  in run.get_metrics().items() if isinstance(v, float)}
    all_metrics[int(properties['iteration'])] = metrics

run_data = pd.DataFrame(all_metrics)
run_data.head(10)



### Getting the best model
best_run, best_model = local_run.get_output()

# predict on test
y_hat = best_model.predict(X_test)



# calculate r2 on the prediction
acc = np.average(y_hat == Y_test)
local_run.log('accuracy', np.float(acc))

os.makedirs('outputs', exist_ok=True)


# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=best_model, filename='outputs/local_auto_ml_model.pkl')


# upload the model file explicitly into artifacts 
local_run.upload_file(name = 'local_auto_ml_model.pkl', path_or_stream = 'outputs/local_auto_ml_model.pkl')

