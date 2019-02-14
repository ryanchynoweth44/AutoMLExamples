## Auto Machine Learning

The traditional development of predictive solutions is challenging and time consuming, that requires expert resources in software development, data engineering, and data science. Typically an engineer is required to complete the following tasks in an iterative and cyclical manner.   
1. Preprocess, feature engineer, and clean data
1. Select appropriate model
1. Tune Hyperparameters 
1. Analyze Results

As the industry identified the blockers that make the development of machine learning solutions costly, we aim to figure out a way to automate the process in an attempt to make it easier and faster to deploy intelligent solutions. Therefore, selecting and tuning models can be automated to make the results analysis easier for non-expert and expert developers.   

Automated machine learning is the ability to have a defined dataset with a specific target feature, and automatically iterate over the dataset with different algorithms and combination of input variables to select the best model. The purpose is to make developing this solutions require less resources, less domain knowledge, and less time. 

### How it works

Most Auto ML libraries available are used to solve supervised learning in order to solve specific problems. If you are unfamiliar, there ar two main categories of machine learning.  
- **Supervised Learning**: is where you have input variables and output variables, and you apply algorithms to learn the mapping function of input to output.    
- **Unsupervised Learning**: is where you have input variables but no output variables to map them to. The goal is typically to identify trends and patterns in the data to make assumptions.  

Note there is a category called **semi-supervised learning** but we will not get into that here.  

In order to use auto machine learning you dataset must be feature engineered. Meaning, you manaully develop transformations to create a machine learning dataset to solve your problem. Most Auto ML libraries have built in transformation functions to solve the most popular tranformation steps, but in my experience these functions are rarely enough to get data machine learning ready. 

Once you have featured engineer your dataset the developer simply needs to determine the type of algorithm they need. Most supervised learning algorithms can be classified as: 

- **Classification**: The output variable is a set number of outcomes. For example, predicting if a customer will return to a store is either a "yes" or a "no". Classification is additionally broken into multiclassification (3 or more outcomes) and binary classification (2 outcomes).  
- **Regression**: The output is a numeric value. For example, predicting the prices of a car or house. 


When given a algorithm type, Auto ML libraries will run iterations over your dataset to determine the best combination features, and best hyperparameters for each algorithm, learn the best algorithm.    


### Available Libraries

[MLBox](https://github.com/AxeldeRomblay/MLBox), a python library for automated machine learning. Key features include distributed processing of data, robust feature selection, accurate hyperparameter tuning, deep learning support, and model interpretation.  

[TransmogrifAI](https://transmogrif.ai/), is an Auto ML library for structured data in Scala purposed for Apache Spark. The library is developed and supported by SalesForce developers.  

[Auto-sklearn](https://automl.github.io/auto-sklearn/stable/), a python library is great for all the sci-kit learn developers out there. It sits on top of sci-kit learn to automate the hyperparameter and algorithm selection process.  

[AzureML](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-automated-ml), an end to end platform for machine learning development and deployment. The library enables faster iterations by manage and tracking experiments, and fully supports most python-based frameworks like PyTorch, Tensorflow, and sci-kit learn. The Auto ML feature is baked into the platform to make it easy to select your model.   


Check out the libraries above! Automated machine learning is fun to play around with and apply to problems. 