## Deploying a Machine Learning Model with Azure Machine Learning

Now that you have automatically trained a machine learning model, it is time to deploy that model as a web service to [Azure Container Instance](https://azure.microsoft.com/en-us/services/container-instances/), the easiest and fastest way to deploy a container in Azure. 



## Setting up the environment
1. We will want to create a new script. Lets name it `deploy_local_model.py` and save it to the application root directory.  

1. Import the require python libraries.  
    ```python
    import pandas as pd
    import numpy as np
    import azureml.core
    from azureml.core import Workspace, Run
    from azureml.core.model import Model
    from azureml.core.authentication import ServicePrincipalAuthentication
    import logging, os
    ```

1. Next we need to connect to our Azure Machine Learning Workspace. We will be able to reuse the config file we created when we deployed our Azure resources.  

    ```python 
    from app_helper import AppHelper
    helper = AppHelper()

    ## Connect to our Azure Machine Learning Workspace
    auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
    ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )
    ```

1. The Azure Machine Learning service expects the model to be in our local (or remote) development workspace to deploy. Therefore, we will need to reference the `.pkl` file in our outputs folder from our training procedure.  
    ```python
    # copy the model to local directory for deployment
    model_name = "local_auto_ml_model.pkl"
    model_path = "outputs/" + model_name
    deploy_folder = os.getcwd()
    ```

1. Since we are deploying a model we will want to first register it with our AML Service workspace.  
    ```python
    # register the model 
    mymodel = Model.register(ws, model_name = model_name, model_path = model_path )
    ```

1. Next we need to write a scoring file to our cluster as well. This is the code that will execute when the web service is called. The following code will write the score string as a python script in your working directory.    
    ```python
    score = """
    import pickle
    import json
    import numpy as np
    import os
    import azureml.train.automl
    from sklearn.externals import joblib
    from azureml.core.model import Model

    def init():    
        global model
        # retreive the path to the model file using the model name
        model_path = Model.get_model_path(model_name  = '{model_name}')
        model = joblib.load(model_path)
        
        
    def run(raw_data):
        try: 
            data = np.array(json.loads(raw_data)['data'])
            
            # make prediction
            y_hat = model.predict(data)
            
            # you can return any data type as long as it is JSON-serializable
            return y_hat.tolist()

        except Exception as e: 
            return str(e)
        
    """.format(model_name=model_name)
    
    exec(score)
    
    with open("web_service_score.py", "w") as file:
        file.write(score)
    ```

1. Next we need to create our config file for deployment. The azure machine learning library has some useful functions to ensure that your `yml` file is in the correct format.   
    ```python
    # Create a dependencies file
    from azureml.core.conda_dependencies import CondaDependencies 

    myenv = CondaDependencies.create(conda_packages=['numpy', 'scikit-learn'], pip_packages=['azureml-sdk[automl]']) #showing how to add libs as an eg. - not needed for this model.

    with open("myenv.yml","w") as f:
        f.write(myenv.serialize_to_string())
    ```

1. We will now configure an Azure Container Instance to deploy to. This will be deployed to our Azure Machine Learning Service Workspace.  
    ```python
    # ACI Configuration
    from azureml.core.webservice import AciWebservice, Webservice

    myaci_config = AciWebservice.deploy_configuration(cpu_cores=1, 
                memory_gb=1, 
                tags={"data": "Titanic",  "method" : "AutoML"}, 
                description='Predict titanic with AutoML')
    ```

1. To deploy a web service with Azure Machine Learning we need to create a Docker Image 
    ```python
    # deploy to aci
    from azureml.core.webservice import Webservice
    from azureml.core.image import ContainerImage

    # configure the image
    image_config = ContainerImage.image_configuration(execution_script="web_service_score.py", 
                            runtime="python", 
                            conda_file="myenv.yml", 
                            description = "Auto ML model",
                            tags = {"data": "titanic", "type": "classification"}
                            )
    ```

1. Now we need to deploy our container to the Azure Container Instance we just deployed as well. 
    ```python
    service = Webservice.deploy_from_model(workspace=ws,
                                        name='automlwebservice',
                                        deployment_config=myaci_config,
                                        models=[mymodel],
                                        image_config=image_config)

    service.wait_for_deployment(show_output=True)
    ```


1. You can print the url of the web service if you wish.  
    ```python
    # print the uri of the web service
    print(service.scoring_uri)
    ```

1. For testing purposes lets load some data and score it against our web service. 
    ```python
    #### test the web service
    import requests
    import json

    # we don't want to send nan to our webservice. Replace with 0. 
    test_data = pd.read_csv("data/titanic_test.csv").fillna(value=0).values
    ```

1. Let's quickly test and make sure our web service is working. Run the following code to see if it works. 
    ```python
   # send a random row from the test set to score
    random_index = np.random.randint(0, len(test_data)-1)
    ## we want to use double quotes in our json
    input_data = "{\"data\": [" + str(list(test_data[random_index])).replace("\'", "\"") + "]}"

    headers = {'Content-Type':'application/json'}

    resp = requests.post(service.scoring_uri, input_data, headers=headers)

    print("POST to url", service.scoring_uri)
    print("label:", test_data[random_index])
    print("prediction:", resp.text)
    ```

1. You have now successfully deployed a machine learning model using the Azure Machine Learning service and its AutoML capabilities! You will likely want to clean up the azure machine learning workspace in order to avoid charges. Navigate to the Azure Portal and find your workspace. Then click on "Deployments" to delete the container we just deployed.  


1. Check out the next portion of the walk through where we [train a model on a remote data science virtual machine](./04_TrainRemotely.md) in Azure. 