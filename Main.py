import os
import glob #For importing all files at once
import pandas as pd #For reading CSV files
import numpy as np #For handling vector calculations

genre = ['rnb','edm_dance','jazz','latin','pop','kids','classical','rock','country','metal']
Trainingset = pd.read_csv('TrainingTrimmed.csv').set_index('id')


def GaussianFilter(data,genre):
    #Function for Gaussianfilter; this builds the Gaussian filter given all songs in training set with a given genre and outputs a 13X12 matrix
    #The first row is 12-dimensional mean vector and the rest is 12X12 covariance matrix of that genre

    temp = []
    #Find all labels of given genre 
    labels = data[data['category'] == genre]
    paths = glob.glob(os.path.join(os.getcwd(),"training\*"))
    for file in paths:
        if os.path.basename(file) in labels.index: #Check whether this song is the genre we are evaluating
            content = pd.read_csv(file,header = None)
            temp.append(content)
    FinalSet = pd.concat(temp, ignore_index=True) #A N by 12 matrix with values from all segments of that genre    
    mean = FinalSet.mean(axis = 0)
    meannp = mean.to_numpy() #This is now a 12-element nparray with all the mean values of a genre

    #To construct covariance matrix:
    cov = FinalSet.to_numpy()
    cov = np.transpose(cov) #Otherwise np would compute a 120K by 120K covariance matrix...
    cov = np.cov(cov)

    #Give the final filter
    filter = np.concatenate((meannp[None,:],cov))
    return filter    
#Find all songs in each genre
GaussianFilter(Trainingset,'rnb')