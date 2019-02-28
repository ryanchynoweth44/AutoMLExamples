import pandas as pd
import numpy as np
from azure.storage.blob import BlockBlobService
import azureml.core
from azureml.core.workspace import Workspace
from azureml.core.experiment import Experiment
from azureml.core.authentication import ServicePrincipalAuthentication
from azureml.core.compute import AmlCompute
from azureml.train.automl import AutoMLConfig
import logging, os, time
from sklearn.model_selection import train_test_split

from app_helper import AppHelper
helper = AppHelper()

## Connect to our Azure Machine Learning Workspace
auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )

## Experiment name and project folder
experiment_name = 'azureautoml'
project_folder = 'remote_automl'
nodes = 2

exp = Experiment(ws, experiment_name)


dsvm_name = 'dlvmaml2' #Name your DSVM
try:
    dsvm_compute = AmlCompute(ws, dsvm_name)
    print('found existing dsvm.')
except:
    print('creating new dsvm.')
    # Below is using a VM of SKU Standard_D2_v2 which is 2 core machine. You can check Azure virtual machines documentation for additional SKUs of VMs.
    dsvm_config = AmlCompute.provisioning_configuration(vm_size = "Standard_NC6", max_nodes=nodes, min_nodes=0)
    dsvm_compute = AmlCompute.create(ws, name = dsvm_name, provisioning_configuration = dsvm_config)
    dsvm_compute.wait_for_completion(show_output = True)


automl_settings = {
    "name": "AutoML_Demo_Experiment_{0}".format(time.time()),
    "iteration_timeout_minutes": 60,
    "iterations": 20,
    "n_cross_validations": 5,
    "primary_metric" : 'accuracy',
    "preprocess" : True,
    "verbosity" : logging.INFO,
    "max_concurrent_iterations": nodes}

## note here that the project folder gets uploaded to our DSVM.
## therefore we must have any extra classes/files in there as well i.e. app_helper.py and app_config.conf
automated_ml_config = AutoMLConfig(task='classification',
                             debug_log='automl_errors.log',
                             path=project_folder,
                             compute_target = dsvm_compute,
                             data_script=project_folder + "/get_data.py",
                             **automl_settings,
                            )


remote_run = exp.submit(automated_ml_config, show_output=True)
# https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-configure-auto-train