## Set up Development Environment

Our first step is to set up development environment. For python development I use the Anaconda Python distribution and create a new virtual environment for . Once it is created use Visual Studio Code (VS Code) to develop. If you are not sure how to create an anaconda environment and use it in VS Code check out this [blog](https://ryansdataspot.com/2019/02/14/anaconda-environments-in-visual-studio-code/) I wrote walking developers through the process.  

### Create Local Environment 
1. Create an Anaconda virtual environment with pandas installed and attach use it has your python interpreter in VS Code. Here is a [link](https://epistasislab.github.io/tpot/installing/) to the official TPOT installation documentation.  
    ```
    conda create -n myenv python=3.6
    ```

1. Next pip install the following. 
    ```
    pip install numpy scipy scikit-learn pandas

    pip install deap update_checker tqdm stopit

    # pywin32 is required for windows
    pip install pywin32

    # optional install for more available models - i did not install for this demo 
    pip install xgboost

    # optional for parallel training - i did not install for this demo 
    pip install dask[delayed] dask-ml

    # tpot install
    pip install tpot 
    ```


