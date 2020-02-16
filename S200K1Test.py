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
    if os.path.basename(file) in labels.index:
        content = pd.read_csv(file,header = None)
        temp.append(content)
        CurrentLabel = labels.loc[os.path.basename(file),'category']
        templabel = [CurrentLabel] * len(content.index)
        labellist += templabel
FinalSet = pd.concat(temp, ignore_index=True)
Trainingset = pd.read_csv('labels.csv').set_index('id')
print('Training data loading done')
paths = glob.glob(os.path.join(os.getcwd(),"test\*"))
#paths = glob.glob(os.path.join(os.getcwd(),"training\*"))
NumTests = len(paths) 
print('The amount of test samples is ',NumTests)
print('Starting classification of test samples')
FinalResults = pd.DataFrame(index = np.arange(0,NumTests),columns = ['id','category'])
FileNumber = 0
NumAvs = 100 #STore the number of averages we want to take
for file in paths:
    testcontent = pd.DataFrame()
    print('Classifying file ',os.path.basename(file))
    TestGenre = ''
    content = pd.read_csv(file, header = None) #Read in the file
    if len(content.index)>200:
        testcontent = content.sample(n=200)
    else:
        testcontent = content
    results = testcontent.apply(kNN_Classifier, axis = 1,args = [FinalSet,1,labellist])
    genres = results.tolist()
    TestGenre = max(set(map(tuple, genres)),key = genres.count)
    GenreResult = ''.join(TestGenre)#Concatenate the resulted genre name into one word
    FinalResults.loc[FileNumber] = [os.path.basename(file)] + [GenreResult]
    FileNumber += 1
    print('Finished',FileNumber, ' files')
FinalResults.to_csv('S200K1Test.csv',index = False)
