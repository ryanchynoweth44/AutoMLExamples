import azureml.core
import pandas as pd
from azureml.core.workspace import Workspace
import logging
import os

from app_helper import AppHelper

helper = AppHelper()

ws = Workspace.get(name=helper.aml_workspace_name, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )
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