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
    #Find all songs of given genre in the training list
    paths = glob.glob(os.path.join(os.getcwd(),"training\*"))
    for file in paths:
        if os.path.basename(file) in labels.index: #Check whether this song is the genre we are evaluating
            content = pd.read_csv(file,header = None)
            temp.append(content)
    FinalSet = pd.concat(temp, ignore_index=True) #A NX12 matrix with values from all segments of that genre    
    mean = FinalSet.mean(axis = 0)
    meannp = mean.to_numpy() #This is now a 12-element nparray with all the mean values of a genre

    #To construct covariance matrix:
    cov = FinalSet.to_numpy()
    cov = np.transpose(cov) #Otherwise np would compute a 120K by 120K covariance matrix...
    cov = np.cov(cov)

    #Give the final filter
    filter = np.concatenate((meannp[None,:],cov))
    return filter    

filter = GaussianFilter(Trainingset,'rnb')


##Extract data from the file
#content = pd.read_csv(file,header = None)
#content = content.to_numpy()

#Extract relevant info from filter
temp = np.split(filter,[1,filter.shape[0]]) #Split back into a 1-D array of means and a 12X12 matrix of covariances
#Note that after splitting, the first array (1-D) is also contained in another array so splitted_array[0][0][0] will give the first mean
inverseCov = np.linalg.inv(temp[1])
mean = temp[0][0]

def UNLL(segment,mean,inVcov):
    #This function takes a vector of a song segment and the mean and inverse of covariance matrix of a genre, then computes the segment's UNLL for this genre
    result = segment - mean
    temp = np.transpose(result)
    temp = np.dot(inVcov,temp)
    result = np.dot(result,temp)
    return result

content = pd.read_csv('00b0c2b6-c448-4c3d-bd9c-0ad5cee771d5',header = None)
results = content.apply(UNLL, axis = 1, args = [mean, inverseCov]) #This applies the function UNLL to each row of data in the song file
print(results)
