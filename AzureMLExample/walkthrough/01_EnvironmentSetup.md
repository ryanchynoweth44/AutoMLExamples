## Set up Development Environment

Our first step is to set up development environment. For python development I use the Anaconda Python distribution and create a new virtual environment for . Once it is created use Visual Studio Code (VS Code) to develop. If you are not sure how to create an anaconda environment and use it in VS Code check out this [blog](https://ryansdataspot.wordpress.com/2019/02/14/anaconda-environments-in-visual-studio-code/) I wrote walking developers through the process.  

### Create Local Environment 
1. Create an Anaconda virtual environment with pandas installed and attach use it has your python interpreter in VS Code. 
    ```
    conda create -n azautoml python=3.7 scikit-learn
    ```

1. Next pip install the following. 
    ```
    pip install azureml-sdk 
    pip install matplotlib
    ```

### Create Azure Resources
1. We will be using the NYC Taxi Tip Dataset. Please download the data [here](https://bit.ly/2Ezp8dH). Save the file to a `data` folder in your application directory.  

1. [Create an Azure Storage Account](https://docs.microsoft.com/en-us/azure/storage/common/storage-quickstart-create-account?toc=%2Fazure%2Fstorage%2Fblobs%2Ftoc.json&tabs=azure-portal) and [create a blob container](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-portal). Then upload the data to the newly created container. 

1. We now need an Azure Machine Learning Workspace in Azure. Follow the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace#create-a-workspace) to do so. 

1. You will need a service principal id and secret in order to connect to your workspace programmatically. Follow the instructions [here](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal#create-an-azure-active-directory-application) in order to create one. Then give it "contributor" access to your subscription (or resource group) by following these [instructions](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal#assign-the-application-to-a-role). 

1. Create an `app_config.conf` file and save it to your application directory. The config file should be of the following format, and please fill in the values as needed. Take note of the [keys required for authentication](https://docs.microsoft.com/en-us/azure/active-directory/develop/howto-create-service-principal-portal#get-values-for-signing-in) that you created during the previous step. 
    ```
    [AZ_AUTO_ML]
    SUBSCRIPTION_ID = <your subscription id>
    AML_WORKSPACE_NAME = <your azure machine learning workspace name>
    AML_RESOURCE_GROUP = <your resource group in azure>
    AML_LOCATION = <the azure region of your aml resource>
    USERNAME = <client id>
    PASSWORD = <client secret>
    TENANT_ID = <tenant id>
    STORAGE_NAME = <azure storage name>
    STORAGE_KEY = <azure storage key>
    AZURE_DATA_CONTAINER = <storage account data container>
    AZURE_DATA_PATH = <path to data file in storage account>
    LOCAL_DATA_PATH = <local path to file>
    ```

1. Once you have created your `app_config.conf` file lets quickly create our `app_helper.py` class. In smaller and public projects like these I will create a class to easily handle my global variables and secrets. This allows me to ignore the ".conf" file type with git so that my secrets are not exposed. Create a python file called `app_helper.py` in your application directory and paste the following code. 
    ```python
    import sys, os
    import json
    import configparser


    class AppHelper(object):
        """
        This class is a helper class. It provides secrets so that I can use a gitignore. 
        """

        def __init__(self, config_file="app_config.conf", env="AZ_AUTO_ML"):
            self.subscription_id = None
            self.aml_workspace_name = None
            self.aml_resource_group = None
            self.aml_location = None
            self.username = None
            self.password = None
            self.tenant_id = None
            self.storage_name = None
            self.storage_key = None
            self.azure_data_container = None
            self.azure_data_path = None
            self.local_data_path = None
            self.set_config(config_file, env)

        def set_config(self, config_file,  env):
            """
            Sets configuration variables for the application
            :param config_file: the path to the configuration file
            :param env: the environment string to parse in config file
            :return None
            """
            config = configparser.RawConfigParser(allow_no_value=True)
            config.read(filenames = [config_file])
                
            ### Setting values here ###
            self.subscription_id = config.get(env, "SUBSCRIPTION_ID")
            self.aml_workspace_name = config.get(env, "AML_WORKSPACE_NAME")
            self.aml_resource_group = config.get(env, "AML_RESOURCE_GROUP")
            self.aml_location = config.get(env, "AML_LOCATION")
            self.username = config.get(env, "USERNAME")
            self.password = config.get(env, "PASSWORD")
            self.tenant_id = config.get(env, "TENANT_ID")
            self.storage_name = config.get(env, "STORAGE_NAME")
            self.storage_key = config.get(env, "STORAGE_KEY")
            self.azure_data_container = config.get(env, "AZURE_DATA_CONTAINER")
            self.azure_data_path = config.get(env, "AZURE_DATA_PATH")
            self.local_data_path = config.get(env, "LOCAL_DATA_PATH")

    ```

Now that you have set up your project, it is time to move onto training your model! Navigate to the [train model](./02_TrainModel.md) steps to continue. 