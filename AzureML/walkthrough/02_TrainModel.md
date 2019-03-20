## Training a Machine Learning Model with AureML's AutoML

In this section of the walk through we will be using our cloud resources to track and train a model locally using the AutoML library of Azure Machine Learning. Please note that Azure Machine Learning does provide remote compute targets to scale the training process and will be covered later in the walk through. 

The dataset that we are using is the popular titanic dataset where we use information about each passenger to predict whether or not they survived the catastrophe. For more information about the dataset check out the [Kaggle Competition](https://www.kaggle.com/c/titanic).  

1. To start we will create a `train_model_local.py` python script in the root directory of your project folder. 

1. Import dependencies for training a model. 
    ```python
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
    ```

1. To start our experiment we will want to connect to our Azure Machine Learning Workspace. This allows us to track and monitor our solutions and models. Note here we are using the service principal id and secret to authenticate with our Azure services.   
    ```python
    ## Connect to our Azure Machine Learning Workspace
    auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
    ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )
    ```

1. Next we will create an experiment for this demo walk through. 
    ```python
    ## Experiment name and project folder
    experiment_name = 'local-auto-ml-demo'
    project_folder = './local-auto-ml-demo'

    exp = Experiment(ws, experiment_name)
    ```

1. Connect to the Titanic Survival dataset. I have my dataset loaded in an Azure Storage Account so that I can access if from anywhere. I will download it locally if it does not exists and read it into a pandas dataframe. Once in a pandas dataframe I will do a 75/25 split of my dataset.   
    ```python
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
    ```

1. We will need to set some of our auto machine learning settings. Note that best practices is to have a third dataset, test, to ensure we did not overtrain our model before deploying. 
    ```python
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
    ```

1. Its time to train our model. Please note that since we are training locally.   
    ```python
    local_run = exp.submit(automated_ml_config, tags={'Category': 'AutoMLExample'}, show_output=True)
    ```

1. You have now trained several machine learning models using Azure Machine Learning's AutoML. Often I will want to analyze the results of each iteration, therefore, I extract the metadata information and convert it to a pandas dataframe.  
    ```python
    runs = list(local_run.get_children())
    all_metrics = {}

    for run in runs:
        properties = run.get_properties()

        metrics = {k: v for k, v  in run.get_metrics().items() if isinstance(v, float)}
        all_metrics[int(properties['iteration'])] = metrics

    run_data = pd.DataFrame(all_metrics)
    run_data.head(10)
    ```

1. However, if you want to just get the best model use the built in function to get the output of our AutoML.  
    ```python
    best_run, best_model = local_run.get_output()
    ```

1. We can even apply our freshly trained model to our test dataset for further evaluation. 
    ```python
    y_hat = best_model.predict(X_test)

    # calculate accuracy on the test data
    acc = np.average(y_hat == Y_test)
    local_run.log('accuracy', np.float(acc))
    ```

1. Once you have identified the model you would like to deploy as a web service, you will need to save it and upload it to your Azure Machine Learning Workspace. 
    ```python
    os.makedirs('outputs', exist_ok=True)

    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(value=best_model, filename='outputs/local_auto_ml_model.pkl')

    # upload the model file explicitly into artifacts 
    local_run.upload_file(name = 'local_auto_ml_model.pkl', path_or_stream = 'outputs/local_auto_ml_model.pkl')
    ```

You have now automatically trained and saved the best machine learning dataset to predict titanic survivors. Move to the next portion of the demo to [deploy this model](./03_DeployModel.md) as a web service in Azure.   