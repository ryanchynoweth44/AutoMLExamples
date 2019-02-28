import pandas as pd
import numpy as np
import azureml.core
from azureml.core import Workspace, Run
from azureml.core.model import Model
from azureml.core.authentication import ServicePrincipalAuthentication


import logging, os

from app_helper import AppHelper
helper = AppHelper()


## Connect to our Azure Machine Learning Workspace
auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )


# copy the model to local directory for deployment
model_name = "local_auto_ml_model.pkl"
model_path = "outputs/" + model_name
deploy_folder = os.getcwd()

# register the model 
mymodel = Model.register(ws, model_name = model_name, model_path = model_path )

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
    return json.dumps({"result":y_hat.tolist()})

  except Exception as e: 
    return json.dumps('Error': str(e) )
    
""".format(model_name=model_name)
 
exec(score)
 
with open("web_service_score.py", "w") as file:
    file.write(score)


# Create a dependencies file
from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies.create(conda_packages=['numpy', 'scikit-learn'], pip_packages=['azureml-sdk[automl]']) #showing how to add libs as an eg. - not needed for this model.

with open("myenv.yml","w") as f:
    f.write(myenv.serialize_to_string())


# ACI Configuration
from azureml.core.webservice import AciWebservice, Webservice

myaci_config = AciWebservice.deploy_configuration(cpu_cores=1, 
             memory_gb=1, 
             tags={"data": "Titanic",  "method" : "AutoML"}, 
             description='Predict titanic with AutoML')


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

service = Webservice.deploy_from_model(workspace=ws,
                                       name='automlwebservice',
                                       deployment_config=myaci_config,
                                       models=[mymodel],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)

# print the uri of the web service
print(service.scoring_uri)



#### test the web service
import requests
import json

test_data = pd.read_csv("data/titanic_test.csv").values
# send a random row from the test set to score
random_index = np.random.randint(0, len(test_data)-1)
input_data = "{\"data\": [" + str(list(test_data[random_index])) + "]}"

headers = {'Content-Type':'application/json'}

resp = requests.post(service.scoring_uri, input_data, headers=headers)

print("POST to url", service.scoring_uri)
print("label:", test_data[random_index])
print("prediction:", resp.text)
