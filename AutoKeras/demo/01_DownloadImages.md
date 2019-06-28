## Getting Data

Deep learning libraries typically provide built in datasets used to build sample neural networks. These datasets are not usually provided in the form that data scientists will encounter in the wild, therefore, I find it more interesting and fun curate a dataset of your own. 

I am almost done with the fast.ai course (which I would totally recommend to any data scientist) and early in the course they provided a great demo on how to create an image dataset from google images on [GitHub](https://github.com/fastai/course-v3/blob/master/nbs/dl1/lesson2-download.ipynb). I will quickly provide an overview of the process here. Please note that fastai has built in functions to download images so we will need to install the appropriate libraries.  

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

1. Now lets download images! Go to [Google Images](https://www.google.com/imghp), and search for specific images you wouldl like to classify. I will be searching for: Airplanes, Train, and Cars. Please note that the curation of your dataset is very important to the overall performance of your neural network, therefore, you will want to ensure that your search query is very specific so that you download the correct images. For example, if you were to search for tigers you may end up with the animal, sport team logos, and cartoons which could effect your desired outcome. 

1. Once you search for images, navigate all the way to the bottom of the Google Image search. There will be a button prompting "Show more results", click the button and continue to navigate past that until the page ends. 

1. Now press CTRL + Shift + J, paste the following Javascript, and press Enter. 
    ```javascript
    urls = Array.from(document.querySelectorAll('.rg_di .rg_meta')).map(el=>JSON.parse(el.textContent).ou);
    window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')))
    ```
    This downloads a single column CSV file containing all of the image URLs. Please note that you will likely need to add the `.csv` file extension. For example, I renamed the file from `download` to `airplanes.csv`, and moved the file to a `data` subfolder in my working directory.  

1. Repeat the previous step for all categories you desire. For example, I completed the previous step a total of 3 times to download airplanes, trains, and cars. 

1. Now lets create a `download_google_images.py` script. 

1. Lets set up our file paths for downloads. Replace the file and folder names as needed to fit the dataset you downloaded.   
    ```python
    from fastai.vision import *

    files = ['data/airplanes.csv', 'data/trains.csv', 'data/cars.csv']
    # output folder location
    folders = ['data/airplane', 'data/train', 'data/car' ] 
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

You now have you very own image dataset! Before we train a model with AutoKeras I want to provide a very simple and short example of how to traditionally [train a model with keras](./02_TrainWithKeras.md). Please note that this model is not a very high performing model due to a small dataset, small network, and likely poorly configured network layers.  