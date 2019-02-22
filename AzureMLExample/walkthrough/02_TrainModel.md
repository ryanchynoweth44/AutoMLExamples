## Training a Machine Learning Model with AureML's AutoML

In this section of the walk through we will be using our cloud resources to track and train a model locally. Please note that Azure Machine Learning does provide remote compute targets to scale the training process and will be covered later in the walk through. 

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
    import logging, os
    from sklearn.model_selection import train_test_split

    from app_helper import AppHelper
    helper = AppHelper()
    ```

1. To start our experiment we will want to connect to our Azure Machine Learning Workspace. This allows us to track and monitor our solutions and models.  
    ```python
    ## Connect to our Azure Machine Learning Workspace
    auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
    ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )
    ```

1. Next we will create an experiment for this demo walk through. 
    ```python
    ## Experiment name and project folder
    experiment_name = 'auto-ml-demo'
    project_folder = './auto-ml-demo'

    exp = Experiment(ws, experiment_name)
    ```

1. Connect to the New York City Taxi Tip dataset. I have my dataset saved to my Azure Storage Account so that I can access if from anywhere. I will download it if it does not exists and read it into a pandas dataframe.  
    ```python
    ## Read data from blob and load into pandas data frame
    blob_account = BlockBlobService(account_name=helper.storage_name, account_key=helper.storage_key)

    if os.path.exists(helper.local_data_path) == False :
        print("Downloading training data.")
        os.makedirs('./data', exist_ok=True)
        blob_account.get_blob_to_path(container_name=helper.azure_data_container, blob_name=helper.azure_data_path, file_path=helper.local_data_path)

    df = pd.read_csv(helper.local_data_path)
    ```

1. We will need to set some of our auto machine learning settings and we will split our training dataset 75/25 to train and validate our model. Note that best practices is to have a third dataset, test, to ensure we did not overtrain our model before deploying.  
    ```python
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
    ```

    Note above we are dropping or selecting tip_amount from our dataset. This is because that is our label column. We must create numpy array's of our feature columns and label column to pass into Azure's AutoML function.    

1. Its time to train our model. Please note that since we are training locally here the process can take a while and is variable depending on the size of machine you are using.  
    ```python
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



    local_run = exp.submit(automated_ml_config, tags="AutoMLExample", show_output=True)
    ```

1. You have now trained several machine learning models using Azure Machine Learning's AutoML. Often I will want to analyze the results of each iteration, therefore, I extract the metadata information and convert it to a pandas dataframe.  
    ```python
    ### convert the runs to a pandas dataframe 
    runs = list(local_run.get_children())
    all_metrics = {}

    for run in runs:
        properties = run.get_properties()

        metrics = {k: v for k, v  in run.get_metrics().items() if isinstance(v, float)}
        all_metrics[int(properties['iteration'])] = metrics

    run_data = pd.DataFrame(all_metrics)
    ```

1. However, if you want to just get the best model use the built in function to get the output of our AutoML.  
    ```python
    best_run, best_model = local_run.get_output()
    ```

1. Once you have identified the model you would like to deploy as a web service, you will need to save it and upload it to your Azure Machine Learning Workspace. 
    ```python
    os.makedirs('outputs', exist_ok=True)

    # note file saved in the outputs folder is automatically uploaded into experiment record
    joblib.dump(value=best_model, filename='outputs/auto_ml_model.pkl')


    # upload the model file explicitly into artifacts 
    local_run.upload_file(name = 'auto_ml_model.pkl', path_or_stream = 'outputs/auto_ml_model.pkl')
    # register the model 
    run.register_model(model_name = 'auto_ml_model.pkl', model_path = 'outputs/auto_ml_model.pkl' )
    ```