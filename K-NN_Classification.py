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
    #dist = Distance.values.tolist()
    #kNN = FindKNN(k,0,dist)
    kNN = np.argpartition(Distance, range(k))[:k]
    for i in kNN:
        genre.append(labels[i])
    TestGenre = max(set(map(tuple, genre)), key = genre.count)
    #print('Finished a segment')   
    return TestGenre


def FindKNN(k,target,array):#Not currently in use
    #This function finds indices the k-nearest neighbours of target in a given array
    #To find the smallest distances, simply find k-nearest neighbours of 0
    
    indices = []
    values = []
    for i in range(k):
        values.append(10000)
        indices.append(0)
    for j in range(len(array)):        
        for p in range(k):
            if abs(values[p] - target) > abs(array[j]-target):               
                indices[p] = j
                values[p] = array[j]
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
for file in paths:
    print('Classifying file ',os.path.basename(file))
    TestGenre = ''
    content = pd.read_csv(file, header = None) #Read in the file
    testcontent = content.sample(n=10)
    results = testcontent.apply(kNN_Classifier, axis = 1, args = [FinalSet,10,labellist])
    genres = results.tolist()
    #results = kNN_Classifier(content,FinalSet,10,labellist)
    TestGenre = max(set(map(tuple, genres)), key = genres.count)
    GenreResult = ''.join(TestGenre)#Concatenate the resulted genre name into one word
    FinalResults.loc[FileNumber] = [os.path.basename(file)] + [GenreResult]
    FileNumber += 1
    print('Finished',FileNumber, ' files')
FinalResults.to_csv('TestTrial.csv',index = False)
