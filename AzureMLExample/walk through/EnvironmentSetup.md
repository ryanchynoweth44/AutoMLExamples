## Set up Development Environment

Our first step is to set up development environment. For python development I use the Anaconda Python distribution and create a new virtual environment for . Once it is created use Visual Studio Code (VS Code) to develop. If you are not sure how to create an anaconda environment and use it in VS Code check out this [blog](https://ryansdataspot.wordpress.com/2019/02/14/anaconda-environments-in-visual-studio-code/) I wrote walking developers through the process.  

1. Create an Anaconda virtual environment and attach use it has your python interpreter in VS Code. 
    ```
    conda create -n azautoml python=3.7
    ```

1. Next pip install the following. 
    ```
    pip install azureml-sdk[automl,notebooks] matplotlib
    ```

1. We now need an Azure Machine Learning Workspace in Azure. Follow the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/service/how-to-manage-workspace#create-a-workspace) to do so. 


1. Create a config file and save it to your application directory. The config file should be of the following format, and please fill in the values as needed.  
    ```
    [AZ_AUTO_ML]
    subscription_id = <your subscription id>
    aml_workspace_name = <your azure machine learning workspace name>
    aml_resource_group = <your resource group in azure>
    aml_location = <the azure region of your aml resource>
    ```

