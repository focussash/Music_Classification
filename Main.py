import os
import glob #For importing all files at once
import pandas as pd #For reading CSV files
import numpy as np #For handling vector calculations
i = 0
genre = ['rnb','edm_dance','jazz','latin','pop','kids','classical','rock','country','metal']
Trainingset = pd.read_csv('TrainingTrimmed.csv').set_index('id')


def GaussianFilter(data,genre):
    #Function for Gaussianfilter; this builds the Gaussian filter given all songs in training set with a given genre and outputs a 13X12 matrix
    #The first row is 12-dimensional mean vector and the rest is 12X12 covariance matrix of that genre
    filter = []
    #Find all labels of given genre 
    labels = data[data[['category']] == genre]
    paths = glob.glob(os.path.join(os.getcwd(),"training\*"))
    for file in paths:
        if os.path.basename(file) in labels.index: #Check whether this song is the genre we are evaluating
            
    return np.array(filter)    
#Find all songs in each genre
