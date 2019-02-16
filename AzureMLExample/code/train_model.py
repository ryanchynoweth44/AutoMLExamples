import pandas as pd
import numpy as np
import azureml.core
from azureml.core.workspace import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication
import logging
import os

from app_helper import AppHelper
helper = AppHelper()

auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )

# choose a name for the run history container in the workspace
experiment_name = 'auto-ml-demo'
# project folder
project_folder = './auto-ml-demo'

output = {}
output['SDK version'] = azureml.core.VERSION
output['Subscription ID'] = ws.subscription_id
output['Workspace'] = ws.name
output['Resource Group'] = ws.resource_group
output['Location'] = ws.location
output['Project Directory'] = project_folder
pd.set_option('display.max_colwidth', -1)
pd.DataFrame(data=output, index=['']).T


### https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-auto-train-models#start

### https://docs.microsoft.com/en-us/azure/machine-learning/service/quickstart-create-workspace-with-python