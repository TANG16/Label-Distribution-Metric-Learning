# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 21:15:10 2016

@author: SYARLAG1
"""
import os
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler

os.chdir('C:/Users/syarlag1/Desktop/Label-Distribution-Metric-Learning/data')

#####################FUNCTIONS THAT WILL BE USED FOR PREPARING DATA###########################################

def createDistributionLabels(targetArray):
    distributionLabel = []
    for entry in targetArray:
        labelVal = {1:0,2:0,3:0,4:0,5:0}#Initialize all 5 labels as 0s
        for rating in labelVal.keys():
            for radRating in entry:
                if rating == radRating:
                    labelVal[rating] += 0.25
        distributionLabel.append(labelVal.values())
    return np.array(distributionLabel)
    

def splitTrainTest(data,Labels,train_percent,random_state, minmax=False):
    random.seed(random_state)
    indexList = range(len(data))
    random.shuffle(indexList)
    trainIndexList = indexList[:int(len(data)*train_percent)]
    testIndexList = indexList[int(len(data)*train_percent):] 
    train, trainLabels, test, testLabels = data[trainIndexList], Labels[trainIndexList], data[testIndexList], Labels[testIndexList]
    if minmax:
        fit = MinMaxScaler.fit(train)
        train = fit.transform(train)
        test = fit.transform(test)
    return train, trainLabels, test, testLabels

def minMaxNorm(X):
    minmax = lambda x: (x-x.min())/(x.max() - x.min())#performing min-max normalization
    return np.apply_along_axis(minmax, 0, X)


def genSimDistRatioMats(data, targetArray, alpha = 1, scale='01', LabelDistribution = True): 
    X = data
    if LabelDistribution: Y = targetArray
    else: Y = createDistributionLabels(targetArray)     
    S = np.zeros(shape=[Y.shape[0], Y.shape[0]])
    D = np.zeros(shape=[X.shape[0], X.shape[0]])
    R = np.zeros(shape=[X.shape[0], X.shape[0]])
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            if i == j: D[i,j],S[i,j],R[i,j] = 1, 0, 0; continue
            if scale == '01': S[i,j] = np.dot(Y[i],Y[j])/(np.linalg.norm(Y[i])*np.linalg.norm(Y[j]))
            if scale == '-11': S[i,j] = -2*np.dot(Y[i],Y[j])/(np.linalg.norm(Y[i])*np.linalg.norm(Y[j])) + 1
            D[i,j] = 1 - np.dot(X[i],X[j])/(np.linalg.norm(X[i])*np.linalg.norm(X[j]))
            if D[i,j] <= 0.0001: D[i,j] = 0.0001
            R[i,j] = S[i,j]/(D[i,j])**alpha
    return X, S, D, R
    
def findKNeighbourhood(S,D,R,k=3):
    neiWeights=np.zeros(shape=S.shape)
    for i in range(S.shape[0]):
        indices = (-R[i]).argsort()[-k:]
        targetDist = np.amax(D[i,indices])
        neiWeights[i]=np.array([R[i,index] if x<=float(targetDist) else 0 for index,x in enumerate(D[i])])
    return neiWeights
    
def createWeightedDistanceMatrices(Nei, X):
   d_ij_S = np.zeros(shape=[X.shape[0],X.shape[0],X.shape[1]])
   for i in range(X.shape[0]):
       for j in range(i+1, X.shape[0]):
           if -Nei[i,j] > 0:
               d_ij_S[i,j,:] = -Nei[i,j]*np.square(np.array(X[i]-X[j]))

   d_ij_D = np.zeros(shape=[X.shape[0],X.shape[0],X.shape[1]])
   for i in range(X.shape[0]):
       for j in range(i+1, X.shape[0]):
           if -Nei[i,j] < 0:
               d_ij_D[i,j,:] =  -Nei[i,j]*np.square(np.array(X[i]-X[j]))           
               
   return d_ij_S, d_ij_D

def XTransform(M, X):
    L = np.linalg.cholesky(M)
    return np.dot(X,L)
    
##################################COST AND CONSTRAINT FUNCTIONS################################################################
#####NOT FULLY DONE YET:
def affineCost(A, d_ij_S, diagnol = True):
    if diagnol: return sum(np.tensordot(d_ij_D, A, ([2],[0])))
    
def constraint(A, d_ij_S, d_ij_D, diagnol = True):
    if diagnol:
        for i in range()
#######        
def affineCost_nonVectorized(A, Nei, diagnol = True):
    cost = 0
    if diagnol: 
        for i in range(Nei.shape[0]):
            for j in range(Nei.shape[1]):
                if -Nei[i,j] > 0:
                    cost += -Nei[i,j]*(np.dot(np.square(X[i]-X[j]),A))

def constraint(A, Nei, d_ij_D, diagnol = True):
    const = []
    for i in range(Nei.shape[0]):
        for j1 in range(Nei.shape[1]):
            if -Nei[i,j1] < 0:
               for j2 in range(Nei.shape[1]):
                   if -Nei[i,j2] > 0:
                       const.append(-Nei[i,j2]*(np.dot(np.square(X[i]-X[j2]),A))+Nei[i,j1]*(np.dot(np.square(X[i]-X[j1]),A)))

#########################DATA PREPARATION##########################################################################
X_data = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1 , usecols = (range(11,76)))
Y_data = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1, usecols = (84,93,102,111)).astype(int)

X_data = np.genfromtxt('./SJAFeatures.csv', delimiter = ',')
Y_data = np.genfromtxt('./SJALabels.csv', delimiter = ',')

X_data = np.genfromtxt('./naturalSceneFeatures.csv', delimiter = ',')
Y_data = np.genfromtxt('./naturalSceneLabels.csv', delimiter = ',')


trainX, trainY, testX, testY = splitTrainTest(X_data,Y_data,0.7,99)

X,S,D,R = genSimDistRatioMats(data = trainX, targetArray = trainY, scale ='01',alpha = 0.5)

cd ..

np.savetxt('./S.csv',S)
np.savetxt('./R.csv', R)

NN3_Ws = findKNeighbourhood(S,D,R,k=3)
np.savetxt('./N.csv',NN3_Ws)
np.savetxt('./X.csv',trainX)
#####Checks:
np.mean(S)

np.std(S)   
count = 0
for i in range(NN3_Ws.shape[0]): #to make verify size of nei
   if len(set(NN3_Ws[i])) <= 3: count += 1
   
for i in range(R.shape[0]):  #to verify if there are nans
    if sum(np.isnan(R[i]))>0: print i

count = 0
for i in range(D.shape[0]): #to verify if there is anything closer than 0.0001
    if np.amin(D[i])<0.0001: print np.amin(D[i]); count += 1

count = 0
for i in range(NN3_Ws.shape[0]): #to make verify size of nei
   if sum(np.array(list(set(NN3_Ws[i]))>0.5))>0: print set(NN3_Ws[i]); count +=1
########

#########################OPTIMIZATION#######################################################################



