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

from app_helper import AppHelper
helper = AppHelper()


## Connect to our Azure Machine Learning Workspace
auth_obj = ServicePrincipalAuthentication(helper.tenant_id, helper.username, helper.password)
ws = Workspace.get(name=helper.aml_workspace_name, auth=auth_obj, subscription_id=helper.subscription_id, resource_group=helper.aml_resource_group )

## Experiment name and project folder
experiment_name = 'local-auto-ml-demo'
project_folder = './local-auto-ml-demo'

exp = Experiment(ws, experiment_name)

## Read data from blob and load into pandas data frame
blob_account = BlockBlobService(account_name=helper.storage_name, account_key=helper.storage_key)

if os.path.exists(helper.local_data_path) == False :
    print("Downloading training data.")
    os.makedirs('./data', exist_ok=True)
    blob_account.get_blob_to_path(container_name=helper.azure_data_container, blob_name=helper.azure_data_path, file_path=helper.local_data_path)

data = dprep.read_csv(helper.local_data_path)
all_cols = dprep.ColumnSelector(term=".*", use_regex=True)

drop_if_all_null = [all_cols, dprep.ColumnRelationship(dprep.ColumnRelationship.ALL)]

useful_cols = ["cost", "distance", "dropoff_datetime", "dropoff_latitude", "dropoff_longitude",
    "passengers", "pickup_datetime", "pickup_latitude", "pickup_longitude", "tip_amount"]

data = data.replace_na(columns=all_cols).drop_nulls(*drop_if_all_null).keep_columns(columns=useful_cols)

decimal_type = dprep.TypeConverter(data_type=dprep.FieldType.DECIMAL)
integer_type = dprep.TypeConverter(data_type=dprep.FieldType.INTEGER)
datetime_type = dprep.TypeConverter(data_type=dprep.FieldType.DATE)

data = data.set_column_types(type_conversions={
    "pickup_longitude": decimal_type,
    "pickup_latitude": decimal_type,
    "dropoff_longitude": decimal_type,
    "dropoff_latitude": decimal_type,
    "cost": decimal_type,
    "distance": decimal_type,
    "passengers": integer_type,
    "pickup_datetime": datetime_type,
    "dropoff_datetime": datetime_type,
    "tip_amount": decimal_type
})

## filter out values that are not in new york 
data = (data
    .drop_nulls(
        columns=["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude"],
        column_relationship=dprep.ColumnRelationship(dprep.ColumnRelationship.ANY)
    )
    .filter(dprep.f_and(
        dprep.col("pickup_longitude") <= -73.72,
        dprep.col("pickup_longitude") >= -74.09,
        dprep.col("pickup_latitude") <= 40.88,
        dprep.col("pickup_latitude") >= 40.53,
        dprep.col("dropoff_longitude") <= -73.72,
        dprep.col("dropoff_longitude") >= -74.09,
        dprep.col("dropoff_latitude") <= 40.88,
        dprep.col("dropoff_latitude") >= 40.53
    )))

# replace 0's with nulls
data = data.replace(columns="distance", find=".00", replace_with=0).fill_nulls("distance", 0)
data = data.to_number(["distance"])

# split datetime into date and time cols
data = (data
    .split_column_by_example(source_column="pickup_datetime")
    .split_column_by_example(source_column="dropoff_datetime"))

# rename our date and time 
data = (data
    .rename_columns(column_pairs={
        "pickup_datetime_1": "pickup_date",
        "pickup_datetime_2": "pickup_time",
        "dropoff_datetime_1": "dropoff_date",
        "dropoff_datetime_2": "dropoff_time"
    }))


## Transform data so that we get more datetime values i.e. day of week, afternoon vs morning etc. 
## note here we are using column by example function
final_df = (data
    .derive_column_by_example(
        source_columns="pickup_date",
        new_column_name="pickup_weekday",
        example_data=[("2009-01-04", "Sunday"), ("2013-08-22", "Thursday")]
    )
    .derive_column_by_example(
        source_columns="dropoff_date",
        new_column_name="dropoff_weekday",
        example_data=[("2013-08-22", "Thursday"), ("2013-11-03", "Sunday")]
    )

    .split_column_by_example(source_column="pickup_time")
    .split_column_by_example(source_column="dropoff_time")
    # The following two calls to split_column_by_example reference the column names generated from the previous two calls.
    .split_column_by_example(source_column="pickup_time_1")
    .split_column_by_example(source_column="dropoff_time_1")
    .drop_columns(columns=[
        "pickup_date", "pickup_time", "dropoff_date", "dropoff_time",
        "pickup_date_1", "dropoff_date_1", "pickup_time_1", "dropoff_time_1"
    ])

    .rename_columns(column_pairs={
        "pickup_date_2": "pickup_month",
        "pickup_date_3": "pickup_monthday",
        "pickup_time_1_1": "pickup_hour",
        "pickup_time_1_2": "pickup_minute",
        "pickup_time_2": "pickup_second",
        "dropoff_date_2": "dropoff_month",
        "dropoff_date_3": "dropoff_monthday",
        "dropoff_time_1_1": "dropoff_hour",
        "dropoff_time_1_2": "dropoff_minute",
        "dropoff_time_2": "dropoff_second"
    }))

final_df.head(5)
final_df = final_df.drop_columns(columns=["pickup_datetime", "dropoff_datetime"])

# infer data types
type_infer = final_df.builders.set_column_types()
type_infer.learn()
type_infer

final_df = type_infer.to_dataflow()



df = final_df.to_pandas_dataframe()
df = df.drop(["dropoff_longitude", "dropoff_latitude", "pickup_latitude", "pickup_longitude"], axis=1)
train, test = train_test_split(df, test_size=.25)
X_train = train.drop('tip_amount', axis=1).values
Y_train = train['tip_amount'].values
X_test = test.drop('tip_amount', axis=1)
Y_test = test['tip_amount'].values


automl_settings = {
    "iteration_timeout_minutes" : 120,
    "iterations" : 1,
    "primary_metric" : 'r2_score',
    "preprocess" : True,
    "verbosity" : logging.INFO,
    "n_cross_validations": 5
}

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



local_run = exp.submit(automated_ml_config, tags={'Category': 'AutoMLExample'}, show_output=True)

### convert the runs to a pandas dataframe  
runs = list(local_run.get_children())
all_metrics = {}

for run in runs:
    properties = run.get_properties()

    metrics = {k: v for k, v  in run.get_metrics().items() if isinstance(v, float)}
    all_metrics[int(properties['iteration'])] = metrics

run_data = pd.DataFrame(all_metrics)


best_run, best_model = local_run.get_output()


# predict on test
y_hat = best_model.predict(X_test)

# # calculate r2 on the prediction
# acc = np.average(y_hat == Y_test)
# run.log('r2', np.float(acc))

os.makedirs('outputs', exist_ok=True)


# note file saved in the outputs folder is automatically uploaded into experiment record
joblib.dump(value=best_model, filename='outputs/auto_ml_model.pkl')


# upload the model file explicitly into artifacts 
local_run.upload_file(name = 'auto_ml_model.pkl', path_or_stream = 'outputs/auto_ml_model.pkl')

