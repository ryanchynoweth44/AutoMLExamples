## Deploying a Machine Learning Model with Azure Machine Learning

Now that you have automatically trained a machine learning model, it is time to deploy that model as a web service to [Azure Container Instance](https://azure.microsoft.com/en-us/services/container-instances/), the easiest and fastest way to deploy a container in Azure. 



## Setting up the environment
1. We will want to create a new script. Lets name it `deploy_local_model.py` and save it to the application root directory.  

1. Import the require python libraries.  
    ```python
    
    ```

1. Next we need to connect to our Azure Machine Learning Workspace. We will be able to reuse the config file we created when we deployed our Azure resources.  

    ```python 
    from app_helper import AppHelper
    helper = AppHelper()

    ## Connect to our Azure Machine Learning Workspace
    auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
    ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )

    ```

1. The Azure Machine Learning service expects the model to be in the working directory. Therefore, we will need to reference the `.pkl` file in our outputs folder from our training procedure.  
    ```python
    
    
    ```

1. Since we are deploying a model we will want to first register it with our AML Service workspace.  
    ```python
    # register the model 
    
    ```

1. Next we need to write a scoring file to our cluster as well. This is the code that will execute when the web service is called.  
    ```python
    
    ```

1. Next we need to create our config file for deployment. 
    ```python
    
    ```

1. We will now configure an Azure Container Instance to deploy to. This will be deployed to our Azure Machine Learning Service Workspace.  
    ```python
    
    ```

1. Now we are ready to actually deploy our model as a web service.  
    ```python
    
    ```

1. Once the image is deployed you can run the following commands to get service logs. This is most useful when the deployment fails. 
    ```python 

    ```

1. You can print the url of the web service if you wish.  
    ```python

    ```

1. For testing purposes lets load some data and score it against our web service. 
    ```python

    ```

1. Let's quickly test and make sure our web service is working. Run the following code to see if it works. 
    ```python
   
    ```

1. You have now successfully deployed a machine learning model using the Azure Machine Learning service and its AutoML capabilities! You will likely want to clean up the azure machine learning workspace in order to avoid charges. Navigate to the Azure Portal and find your workspace. Then click on "Deployments" to delete the container we just deployed.  
