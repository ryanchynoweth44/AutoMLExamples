## Auto Machine Learning with Azure Machine Learning

I recently wrote a blog discussing the pros and cons of using automated machine learning (Auto ML) libraries when developing predictive solutions. If you have not read it you can check it out [here](https://github.com/ryanchynoweth44/AutoMLExamples/blogs/AutoMachineLearning.md). With there being a surplus of Auto ML libraries in the marketplace my goal is to provide an overview and demo of libraries that I use to develop solutions.  

### Using Azure Machine Learning

An Azure Machine Learning Workspace (AML Workspace) a foundational resource for tracking experiments where developers are developing, training, and deploying machine learning solutions as web services. When an engineer provisions an Azure Machine Learning Workspace the resources below are also created within the same resource group. The resources essentially power Azure Machine Learning:
- Azure Container Registry
- Azure Storage 
- Azure Application Insights
- Azure Key Vault


The Azure Container Registry gives developer easy integration with creating, storing, and deploying our Python web services as Docker Containers. One added future is the easy and automatic tagging to describe your container and associate the container with specific machine learning models. 

Our Azure Storage account enables for fast dynamic storing of information from our experiments i.e. models, outputs. After training an inital model using the service, I would recommend manually navigating through the folders. Doing this will give you deeper insight into how the AML Workspace functions. 

When we deploy a web service using the AML Service, we allow the Azure Machine Learning resource to handle all authentication and key generation code. This allows data scientists to focus on developing models instead of writing authentication code. Using Azure Key Vault, the AML Service allows for extremely secure web services that you can expose to external and internal customers.  

Once your secure web service is deployed. Azure Machine Learning integrates seamlessly Application Insights for all code logging and web service traffic giving users the ability to monitor the health of the deployed solution. 

A key future to allowing data scientists to scale their solutions is offering remote compute targets. Remote compute gives developers the ability easily get their solution off their laptop and into Azure with a familiar IDE and workflow. 

Many data platforms offer specialized "pipeline" functions and classes the allow developers to package their data transformations into a single line of code for model deployment. Azure Machine Learning calls this "dprep" or a data prep file. This is an easy way to handle the required data transformation to score new data in production.  

In addition to remote compute, Azure Machine Learning enables users to deploy anywhere they can run docker. Theoretically, one could train a model locally and deploy a model local (or another cloud), and only simply use Azure to track their experiments for a cheap monthly rate. However, I would suggest tracking advantage of Azure Kubernetes Service for auto scaling of your web servie to handle up ticks in traffic, or to a more consistent compute targe in Azure Container Instance. 

## Using Azure Machine Learning's AutoML

In order to use Azure Machine Learning's AutoML capabilities you will need to pip install `azureml-sdk`. This is the same Python library used to simply track your experiments in the cloud.  

As with any data science project, it starts with data aquistion and exploration. In this phase of developing we are exploring our dataset and identifying desired feature columns to use to make predictions. Our goal here is to create a machine learning dataset to predict our label column. 

Once we have created our machine learning dataset and identified if we going to implement a classification or a regression solution, we can let Azure Machine Learning do the rest of the work to identify the best feature column combination, algorithm, and hyper-parameters. 

--------------------------------
A  
B  
C  
D  

MORE SPECFICS HERE ON WHAT HAPPENS BEHIND THE SCENES OR TO LAUNCH AND EXPERIMENT    

E  
F  
G  
H  

-------------------------------------

The output of this process is a dataset containing metadata the training runs and their results. This dataset enables devlopers to easily choose the best model based off the metrics provided. Being able to choose the best model out of many training iterations with different algorithms and feature columns automatically is that it enables us to easily automate the model selection process for *each* model deployment. With typical machine learning deployments, engineers typically deploy the same algorithm with the same feature columns each time. But with Auto Machine Learning solutions we are able to note only choose the best algorithm, feature combination, and hyper-parameters each time. That means, we can deploy a decision tree model trained on 4 columns one release, the deploy a logistic regression model trained on 5 columns another release without any code edits. 