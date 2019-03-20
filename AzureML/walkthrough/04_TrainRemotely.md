## Training a Machine Learning Model with AureML's AutoML

In this section of the walk through we will be using our cloud resources to track and train a model using remote compute targets in Azure ML. We will use the AutoML library of Azure Machine Learning to train a model. 

The dataset that we are using is the popular titanic dataset where we use information about each passenger to predict whether or not they survived the catastrophe. For more information about the dataset check out the [Kaggle Competition](https://www.kaggle.com/c/titanic).  

1. To start we will create a `train_model_remote.py` python script in the root directory of your project folder. 

1. Now let's import the required dependencies.  
    ```python
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
    from sklearn.externals import joblib

    from app_helper import AppHelper
    helper = AppHelper()
    ```

1. We need to authenticate against our Azure Machine Learning Workspace so that we can track our training and execute on a VM in Azure.  
    ```python
    ## Connect to our Azure Machine Learning Workspace
    auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
    ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )
    ```

1. Now we want to simply need to set a few variables to use later on in our experiment script.  
    ```python
    ## Experiment name and project folder, and max nodes for remote compute
    experiment_name = 'azureautoml'
    project_folder = 'remote_automl'
    nodes = 4 

    # create/connect to our ML experiment
    exp = Experiment(ws, experiment_name)
    ```

1. With remote compute targets we need to set a `project folder` as seen above. This project folder needs to contain any other scripts required to execute the experiment, including a `get_data.py` file that allows the remote compute target to acquire the data. In the project folder, `remote_automl`, create a `get_data.py` file and paste the following code: 
    ```python
    from azure.storage.blob import BlockBlobService
    import os
    import pandas as pd
    import numpy as np
    from io import StringIO
    from app_helper import AppHelper
    helper = AppHelper()

    def get_data():
        ## Read data from blob and load into pandas data frame
        print("---------------- Connecting to blob")
        blob_account = BlockBlobService(account_name=helper.storage_name, account_key=helper.storage_key)
        
        print("---------------- reading blob to bytes")
        bytes_data = blob_account.get_blob_to_bytes(container_name=helper.azure_data_container, blob_name=helper.azure_data_path)

        print("---------------- Connecting to blob")
        data = StringIO(bytes_data.content.decode('utf-8')) 

        print("---------------- Reading data")
        df = pd.read_csv(data)

        X_df = df.drop(['Survived'], axis=1)
        Y_df = df['Survived'].values
        return { "X" : X_df, "y" : Y_df }
    ```

1. In addition to the `get_data.py` file we will want to copy and paste the `app_helper.py` and `app_config.conf` files into the same project folder. We created these files in the [Environment Setup](./01_EnvironmentSetup.md) portion of the demo.  

1. We now want to create a remote compute target in our Azure ML Workspace. Azure ML's remote compute allows for parallel training of iterations using queues to execute jobs on different nodes. The cluster is set to have a min and max number of nodes allowing the developer to only pay for what they use. We will create a cluster with a min of 0 and a max of 4. More information about the remote compute can be found [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-auto-train-remote).  
    ```python
    dsvm_name = 'dsvmaml'
    try:
        dsvm_compute = AmlCompute(ws, dsvm_name)
        print('found existing dsvm.')
    except:
        print('creating new dsvm.')
        # Below is using a VM of SKU Standard_D2_v2 which is 2 core machine. You can check Azure virtual machines documentation for additional SKUs of VMs.
        dsvm_config = AmlCompute.provisioning_configuration(vm_size = "Standard_NC6", max_nodes=nodes, min_nodes=0)
        dsvm_compute = AmlCompute.create(ws, name = dsvm_name, provisioning_configuration = dsvm_config)
        dsvm_compute.wait_for_completion(show_output = True)
    ```


1. We will want to set our automl settings for our auto ml jobs.   
    ```python
    automl_settings = {
        "name": "AutoML_Demo_Experiment_{0}".format(time.time()),
        "iteration_timeout_minutes": 60,
        "iterations": 20,
        "n_cross_validations": 5,
        "primary_metric" : 'accuracy',
        "preprocess" : True,
        "verbosity" : logging.INFO,
        "max_concurrent_iterations": nodes}
    ```

1. We now need to create an AutoMLConfig object that contains all the information needed to execute our job.  
    ```python
    ## note here that the project folder gets uploaded to our DSVM.
    ## therefore we must have any extra classes/files in there as well i.e. app_helper.py and app_config.conf
    automated_ml_config = AutoMLConfig(task='classification',
                debug_log='automl_errors.log',
                path=project_folder,
                compute_target = dsvm_compute,
                data_script=project_folder + "/get_data.py",
                **automl_settings,
            )
    ```

1. Now we can submit our experiment to the remote compute target. 
    ```python
    remote_run = exp.submit(automated_ml_config, show_output=True)
    ```

1. Similar to training our model locally we want to get the best model, save it locally, and upload it to our Azure ML Workspace. 
    ```python
    ### Getting the best model
    best_run, best_model = remote_run.get_output()

    os.makedirs('outputs', exist_ok=True)

    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(value=best_model, filename='outputs/remote_auto_ml_model.pkl')

    # upload the model file explicitly into artifacts 
    remote_run.upload_file(name = 'remote_auto_ml_model.pkl', path_or_stream = 'outputs/remote_auto_ml_model.pkl')
    ```

Remote compute targets are an easy way to get predictive workloads off a data scientist's laptop and into the cloud. The parallel training allows the developer to train multiple models (even without AutoML) at a single time to allow for faster feedback and shorten the time to market. At this point we have trained a model remotely and it is available in our workspace, if we wish to deploy then it will be the same process as [deploying](./03_DeployModel.md) the locally trained model. 