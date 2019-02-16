## Training a Machine Learning Model with AureML's AutoML



1. Import dependencies for training a model. 
    ```python
    import pandas as pd
    import numpy as np
    import azureml.core
    from azureml.core.workspace import Workspace
    from azureml.core.authentication import ServicePrincipalAuthentication
    import logging
    import os

    # custom class for secrets
    from app_helper import AppHelper
    # initialize class
    helper = AppHelper()
    ```

1. To start our experiment we will want to connect to our Azure Machine Learning Workspace. This allows us to track and monitor our solutions and models.  
    ```python
    auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
    ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )
    ```


