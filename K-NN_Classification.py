import os
import glob #For importing all files at once
import pandas as pd #For reading CSV files
import numpy as np #For handling vector calculations
from heapq import nsmallest
import profile


def kNN_Classifier(segment,Model,k,labels):
    #This function takes one segment vector from a song and calculate its distance from elemts of the whole model; 
    #then, it finds its k nearest neighbours and assigns a final classification for that segment according to the labels
    genre = []
    Difference = np.square(Model - segment)
    Distance = np.sqrt(np.sum(Difference,axis = 1))
    kNN = FindKNN(k,0,Distance)
    for i in kNN:
        genre.append(labels[i])
    return genre


def FindKNN(k,target,array):
    #This function finds indices the k-nearest neighbours of target in a given array
    #To find the smallest distances, simply find k-nearest neighbours of 0
    indices = []
    for i in range(k):
        indices.append(100000)
    for j in range(len(array)):
        for i in range(k):
            if abs(indices[i] - target) > abs(array[j]-target):
                indices[i] = j
                break
    return indices

#First, concatanate all training data to a big array
temp = []
paths = glob.glob(os.path.join(os.getcwd(),"training\*"))
print('The amount of training samples is',len(paths))
print('Loading training samples')
labels = pd.read_csv('labels.csv').set_index('id')
labellist = []


for file in paths:
    content = pd.read_csv(file,header = None)
    temp.append(content)
    CurrentLabel = labels.loc[os.path.basename(file),'category']
    templabel = [CurrentLabel] * len(content.index)
    labellist += templabel
FinalSet = pd.concat(temp, ignore_index=True)
print('Training data loading done')
paths = glob.glob(os.path.join(os.getcwd(),"test\*"))
NumTests = len(paths) 
print('The amount of test samples is ',NumTests)
print('Starting classification of test samples')
FinalResults = pd.DataFrame(index = np.arange(0,NumTests),columns = ['id','category'])
FileNumber = 0
pausepoint = NumTests//5
for file in paths[1:2]:
    print('Classifying file ',os.path.basename(file))
    TestGenre = ''
    content = pd.read_csv(file, header = None) #Read in the file
    results = kNN_Classifier(content,FinalSet,3,labellist)
    TestGenre = max(set(results), key = results.count) 
    FinalResults.loc[FileNumber] = [os.path.basename(file)] + [TestGenre]
    FileNumber += 1
    print('Finished',FileNumber, ' files')
FinalResults.to_csv('TestTrial.csv', header=False,index = False,mode = 'a')