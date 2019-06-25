# Setting up Anaconda Environment

Please complete the following instructions to setup an Anaconda Environment for AutoKeras development. Please note that the [Anaconda](https://anaconda.com) Python distribution must be installed in order to complete the steps below.    

1. Create a conda environment running Python 3.6.
    ```
    conda create -n autokerasenv python=3.6

    conda activate autokerasenv
    ```


1. To develop using AutoKeras I would recommend using a Linux machine so that you can easily run `pip install autokeras` to get started. 

1. I would also recommend install the `keras` and `tensorflow` libraries as well so that we can utilize additional functions, and we will be training a model using the `keras` prior to training with `autokeras`. 
    ```
    pip install keras

    pip install tensorflow

    pip install pillow

    ## gpu version of tensorflow
    pip install tensorflow-gpu
    ```


## Getting Data

When learning a new deep learning library I always find it a little more interesting when I curate a dataset of my own. While taking the fast.ai course they provide a great demo on how to create an image dataset from google images on [GitHub](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb). I will quickly provide an overview of the process here. Please note that fastai has built in functions to download images so we will need to install the appropriate libraries.  

1. Lets create a new anaconda environment for our image downloads. 
    ```
    conda create -n fastai python=3.6 -y

    conda activate fastai
    ```

1. First we need to install PyTorch. For a windows 10 machine with a CPU run the following. If you are following along on a separate machine check out the [PyTorch webpage on getting started](https://pytorch.org/get-started/locally/#anaconda).  
    ```
    conda install pytorch-cpu torchvision-cpu -c pytorch
    ```

1. Then install fastai.  
    ```
    pip install fastai
    ```

1. Now lets download images! Go to [Google Images](https://www.google.com/imghp), and search for specific images to train on. Good examples are always different types of animals, but you can choose anything you want. I will be searching for: Airplanes, Train, and Cars. Please note that the curation of your dataset is very important to the overall performance of your neural network, therefore, you will want to ensure that your search query is very specific so that you download the correct images. For example, if you were to search for tigers you may end up with the animal, sport team logos, cartoons etc. 

1. Once you search for images, navigate all the way to the bottom of the Google Image search. There will be a button prompting "Show more results", click the button and continue to navigate past that until the page ends. 

1. Now press CTRL + Shift + J, paste the following Javascript, and press Enter. 
    ```javascript
    urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
    window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')))
    ```
    This downloads a single column CSV file containing all of the image URLs. Please note that you will likely need to add the `.csv` file extension. For example, I renamed the file from `download` to `airplanes.csv`, and moved the file to a `data` subfolder in my working directory.  

1. Repeat the previous step for all categories you desire. For example, I completed the previous step a total of 3 times to download airplanes, trains, and cars. 

1. Now lets create a `download_google_images.py` script. 

1. Lets set up our file paths for downloads. 
    ```python
    from fastai.vision import *

    files = ['data/airplanes.csv', 'data/trains.csv', 'data/cars.csv']
    folders = ['data/airplane', 'data/train', 'data/car' ] # output folder location
    ```

1. Now we download images in our csv files, and output them to the folder we want. 
    ```python
    n = len(files)

    for i in range(0,n):
        os.makedirs(folders[i], exist_ok=True)
        download_images(files[i], folders[i], max_pics=500)
    ```

1. FastAI has another function to verify that we have downloaded images successfully i.e. some images may be corrupted.  
    ```python
    for i in range(0,n):
        print("Check for valid images in folder: '" + folders[i] + "'" )
        verify_images(folders[i], delete=True, max_size=500)
    ```

You now have you very own image dataset!