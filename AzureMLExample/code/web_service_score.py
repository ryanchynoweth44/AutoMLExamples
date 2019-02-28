
 
import json
import numpy as np
import os
import pickle
import azureml.train.automl
from sklearn.externals import joblib
from azureml.core.model import Model
 
def init():    
  global model
  # retreive the path to the model file using the model name
  model_path = Model.get_model_path('local_auto_ml_model.pkl')
  model = joblib.load(model_path)
    
    
def run(raw_data):
  data = np.array(json.loads(raw_data)['data'])
  
  # make prediction
  y_hat = model.predict(data)
  
  # you can return any data type as long as it is JSON-serializable
  return y_hat.tolist()
    
