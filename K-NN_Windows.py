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
    kNN = np.argpartition(Distance, range(k))[:k]
    for i in kNN:
        genre.append(labels[i])
    TestGenre = max(set(map(tuple, genre)), key = genre.count)
    #print('Finished a segment')   
    return TestGenre

#First, concatanate all training data to a big array
temp = []
paths = glob.glob(os.path.join(os.getcwd(),"training\*"))
#Change to the following one if the previous path gives an error. Typically, this is caused by different operating systems
#paths = glob.glob(os.path.join(os.getcwd(),"training/*")) 
print('The amount of training samples is',len(paths))
print('Loading training samples')
labels = pd.read_csv('labels.csv').set_index('id')
labellist = [] #We need to also concatenate an array for the labels of data
for file in paths:
    if os.path.basename(file) in labels.index:
        content = pd.read_csv(file,header = None)
        temp.append(content)
        CurrentLabel = labels.loc[os.path.basename(file),'category']
        templabel = [CurrentLabel] * len(content.index) #For each segment introduced in the data array, we add its label to label array
        labellist += templabel
FinalSet = pd.concat(temp, ignore_index=True)
print('Training data loading done')
paths = glob.glob(os.path.join(os.getcwd(),"test\*")) #Now, load testing data
#Change to the following one if the previous path gives an error. Typically, this is caused by different operating systems
#paths = glob.glob(os.path.join(os.getcwd(),"test/*")) 
NumTests = len(paths) 
print('The amount of test samples is ',NumTests)
print('Starting classification of test samples')
FinalResults = pd.DataFrame(index = np.arange(0,NumTests),columns = ['id','category']) #Initialize a dataframe to store the outputs
FileNumber = 0
for file in paths:
    testcontent = pd.DataFrame()
    print('Classifying file ',os.path.basename(file))
    TestGenre = ''
    content = pd.read_csv(file, header = None) #Read in the file
    if len(content.index)>100: #If there is no more than 100 segments in a song file, just take all segments
        testcontent = content.sample(n=100) #Otherwise, randomly sample 100 from the file (as shown in report, this is representative enough)
    else:
        testcontent = content
    results = testcontent.apply(kNN_Classifier, axis = 1,args = [FinalSet,1,labellist]) #Apply KNN classifier to each row, which is one segment
    genres = results.tolist()
    TestGenre = max(set(map(tuple, genres)),key = genres.count) #For each song file, find the most common genres of all its segments
    GenreResult = ''.join(TestGenre)#Concatenate the resulted genre name into one word
    FinalResults.loc[FileNumber] = [os.path.basename(file)] + [GenreResult]
    FileNumber += 1
    print('Finished',FileNumber, ' files')
FinalResults.to_csv('Output.csv',index = False)
