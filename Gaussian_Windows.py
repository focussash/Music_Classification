import os
import glob #For importing all files at once
import pandas as pd #For reading CSV files
import numpy as np #For handling vector calculations

genre = ['rnb','edm_dance','jazz','latin','pop','kids','classical','rock','country','metal']


def GaussianFilter(data,genre):
    #Function for Gaussianfilter; this builds the Gaussian filter given all songs in training set with a given genre and outputs a 13X12 matrix
    #The first row is 12-dimensional mean vector and the rest is 12X12 covariance matrix of that genre

    temp = []
    #Find all labels of given genre 
    labels = data[data['category'] == genre]
    #Find all songs of given genre in the training list
    paths = glob.glob(os.path.join(os.getcwd(),"training\*"))
    #Change to the following one if the previous path gives an error. Typically, this is caused by different operating systems
    #paths = glob.glob(os.path.join(os.getcwd(),"training/*")) 
    for file in paths:
        if os.path.basename(file) in labels.index: #Check whether this song is the genre we are evaluating
            content = pd.read_csv(file,header = None)
            temp.append(content)
    FinalSet = pd.concat(temp, ignore_index=True) #A NX12 matrix with values from all segments of that genre    
    mean = FinalSet.mean(axis = 0)
    meannp = mean.to_numpy() #This is now a 12-element nparray with all the mean values of a genre

    #To construct covariance matrix:
    cov = FinalSet.to_numpy()
    cov = np.transpose(cov) #Otherwise np would compute a 1 million by 1 million covariance matrix...
    cov = np.cov(cov)

    #Give the final filter
    filter = np.concatenate((meannp[None,:],cov))
    return filter    

def UNLL(segment,mean,inVcov):
    #This function takes a vector of a song segment and the mean and inverse of covariance matrix of a genre, then computes the segment's UNLL for this genre
    result = segment - mean
    temp = np.transpose(result)
    temp = np.dot(inVcov,temp)
    result = np.dot(result,temp)
    return result

Trainingset = pd.read_csv('labels.csv').set_index('id')
FileNumber = 0
paths = glob.glob(os.path.join(os.getcwd(),"test\*"))
#Change to the following one if the previous path gives an error. Typically, this is caused by different operating systems
#paths = glob.glob(os.path.join(os.getcwd(),"test/*")) 
NumTests = len(paths) 
FinalResults = pd.DataFrame(index = np.arange(0,NumTests),columns = ['id','category'])

#First, build all filters
InvArray = []
MeanArray = []
print('Building classifier')
for i in genre: 
        filter = GaussianFilter(Trainingset,i)
        #Extract relevant info from filter
        temp = np.split(filter,[1,filter.shape[0]]) #Split back into a 1-D array of means and a 12X12 matrix of covariances
        #Note that after splitting, the first array (1-D) is also contained in another array so splitted_array[0][0][0] will give the first mean
        inverseCov = np.linalg.inv(temp[1])
        meanV = temp[0][0]
        InvArray.append(inverseCov)
        MeanArray.append(meanV)
print('Finished building classifier')
#Now, classify each song
for file in paths:
    print('Classifying file ',os.path.basename(file))
    TestUNLL = 10000000
    TestGenre = ''
    GenreNum = 0
    #if os.path.basename(file) in Testset.index: #Check whether this song is the genre we are evaluating
    TestContent = pd.read_csv(file, header = None) #Read in the file
    #Now, for each genre we build the filter and test the file on filter to extract results
    for i in genre: 
        TestResults = TestContent.apply(UNLL, axis = 1, args = [MeanArray[GenreNum], InvArray[GenreNum]]) #This applies the function UNLL to each row of data in the song file
        if TestUNLL >TestResults.mean():
            TestUNLL = TestResults.mean()
            TestGenre = i
        GenreNum += 1
    #Now, TestGenre will be the genre of this song
    FinalResults.loc[FileNumber] = [os.path.basename(file)] + [TestGenre]
    FileNumber += 1
    print('Finished',FileNumber, ' files')
FinalResults.to_csv('Output.csv',index = False)
