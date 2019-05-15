# Microsoft Research's InterpretML

Recently a team at Microsoft Research has released a Python machine learning library called [InterpretML](https://github.com/microsoft/interpret), with the goal of being able to explain blackbox and interpretable machine learning models. As you can read on the [Github README](https://github.com/microsoft/interpret/blob/master/README.md), data scientists have historically struggled to provide answers to why their models predicted specific values or what features are most important to the outcome. InterpretML makes this easily possible by using an Explainable Boosting Machine, and supports LIME, SHAP, linear models, partial dependence, decision trees and rule lists as well. The team that developed the library wrote a [paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/KDD2015FinalDraftIntelligibleModels4HealthCare_igt143e-caruanaA.pdf) explaining not only the algorithm, but provide great use cases highlighting the problems we face as predictive analytics developers and our in ability to deploy the best model if we are unable to explain them. A caution in the paper that I find important to add here is that, while this libary can provide explainability to a model, it does not provide actual causation to an outcome. InterpretML simply provides users the ability to know the correlation between data points and the prediction so that we can then raise questions as to if or what needs further investigating so we deploy an fair and unbiased model. 

Initially, I believe InterpretML was a library that simply interpretted models and told us what features were most critical. After some initial development of the library I realized they provide a number of machine learning algorithms as well. As of now these models are restricted to binary classification and regression problems, however, multiclassification is on their roadmap. While this library is under its alpha release, it works extremely well with other popular python libraries like pandas, numpy, and scikit learn. While I have not implemented a workflow, I believe that using the AutoML capabilities of Azure Machine Learning or Auto sklearn in union with InterpretML. It should be easy to identify and train a model using the Auto ML libraries, then translate the best features and hyperparameters into an InterpretML model for further analysis.  

Per InterpretML's Github page, it is necessary to explain our models in order to:

- Debug models but understanding why and how our model made a specific mistake
- Detect bias in order to avoid discriminating against specifics groups
- Human cooperation with our models so that we can understand why the model predicted what it did
- Compliance to satisfy any legal requirements
- High risk applications to be reassured of model accuracy 


## Using InterpretML

### Development Environment 

1. Open an Anaconda Command Prompt, and create an Python 3.6 Anaconda Environment 
    ```
    conda create -n interpretmlenv python=3.6
    conda activate interpretmlenv
    ```

1. Install the following libraries. 
    ```
    # required
    pip install numpy scipy pyscaffold
    pip install -U interpret
    ```
