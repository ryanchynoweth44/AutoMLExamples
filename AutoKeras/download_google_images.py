from fastai.vision import *
import os

files = ['data/airplanes.csv', 'data/trains.csv', 'data/cars.csv']
folders = ['data/airplane', 'data/train', 'data/car' ]
n = len(files)

for i in range(0,n):
    print("------- Downloading: " + files[i])
    os.makedirs(folders[i], exist_ok=True)
    download_images(files[i], folders[i], max_pics=500)
    

for i in range(0,n):
    print("Check for valid images in folder: '" + folders[i] + "'" )
    verify_images(folders[i], delete=True, max_size=500)