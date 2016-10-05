# -*- coding: utf-8 -*-
"""
Created on Tue Oct 04 23:03:51 2016

@author: SYARLAG1
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import MinMaxScaler

os.chdir('C:/Users/syarlag1/Desktop/Label-Distribution-Metric-Learning')
#os.chdir('/Users/Sriram/Desktop/DePaul/Label-Distribution-Metric-Learning/data')

from metricLearningFunctions import *

###################Initial data Exploration####################################
metrics = ['cosine', 'fidelity', 'intersection', 'euclidean', 'sorensen', 'squaredChiSq',\
               'chebyshev', 'clark', 'canberra','KL', 'gaussian']


#trial = np.array([[0.1, 0.2, 0.3, 0.4],[0.5, 0.5, 0, 0], [0.1, 0.3, 0.6, 0], [0.2, 0.4, 0.1, 0.3]])
#metricStats(metrics, trial)

os.chdir('./data')

labelsList = []
for fileName in os.listdir('./'):
    if 'Label' in fileName:
        locals()['{0}'.format(fileName)] = np.genfromtxt(fileName, delimiter=',') #This is probably not safe to use
        labelsList.append(fileName)
labelsList

smallerLabelsList = ['SJALabels.csv','naturalSceneLabels.csv', 'YeastSPOEMLabels.csv', 'YeastHeatLabels.csv', 'YeastSPOEMLabels.csv' ]
metricLst = ['cosine', 'fidelity','intersection','euclidean','squaredChiSq','chebyshev']

results = metricStatsforLabelList(metrics, smallerLabelsList)

# To create a gaussian matrices and stats
for filename in smallerLabelsList:
    S, EuclStddev =  gaussSimMatrix(filename, sigma=None)
    print 'sigma = ', EuclStddev, \
    'mean =', np.nanmean(S), 'stddev =', np.nanstd(S),\
    'max =', np.nanmax(S), 'min =', np.nanmin(S)


####Plotting the histograms:

###TEMP:
def genSimDistMat(measure, labels, sigma=None, labelDistribution = True, percentile=True): 
    if type(labels) == str: Y = globals()[labels]
    if ~labelDistribution: Y = createDistributionLabels(Y)   
    S = np.zeros(shape=[Y.shape[0], Y.shape[0]])
    #if measure == 'gaussian': return gaussSimMatrix(labels, sigma)[0]
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            if measure == 'cosine': S[i,j] = np.dot(Y[i],Y[j])/(np.linalg.norm(Y[i])*np.linalg.norm(Y[j]))
            if measure == 'fidelity': S[i,j] = np.sum(np.sqrt(np.multiply(Y[i],Y[j])))
            if measure == 'intersection': S[i,j] =  np.sum(np.minimum(Y[i],Y[j]))   
            if measure == 'euclidean': S[i,j] = np.sqrt(np.dot((Y[i]-Y[j]),(Y[i]-Y[j])))
            if measure == 'sorensen': S[i,j] = np.sum(np.abs(Y[i]-Y[j]))/(np.sum(Y[i]+Y[j])) 
            if measure == 'squaredChiSq': S[i,j] = np.sum(np.square(Y[i]-Y[j])/(Y[i]+Y[j]))
            if measure == 'chebyshev': S[i,j] = np.max(np.abs(Y[i] - Y[j]))            
            if measure == 'clark': S[i,j] =  np.sqrt(np.sum(np.square(Y[i]-Y[j])/np.square((Y[i]+Y[j]))))
            if measure == 'canberra': S[i,j] = np.sum(np.abs(Y[i]-Y[j])/((Y[i]+Y[j])))
            if measure == 'KL':
                tempSum = 0
                for k in range(len(Y[i])):
                    tempSum += Y[i,k]*(np.log((Y[i,k] if Y[i,k]>0 else 0.01)/(Y[j,k] if Y[j,k]>0 else 0.01)))
                S[i,j] = tempSum
    if percentile:
        return convertMatToPercentile(S)
    return S

def convertMatToPercentile(S):
    originalShape = S.shape
    arr = S.flatten()
    sortedUniqueArr = np.unique(np.sort(arr))
    percentileArr = np.linspace(0.01,0.99,len(sortedUniqueArr))
    percentileValueDict = {x:percentileArr[i] for i,x in enumerate(sortedUniqueArr)}
    percentileArr = np.array([percentileValueDict[x] for x in arr])
    
    return np.reshape(percentileArr, originalShape)
    

def histCreator(metricList, labelsList):
    for metric in metricList:
        for label in labelsList:
            figName = metric + '-' + label + '.png'
            simDistArray = np.asarray(genSimDistMat(metric, label, sigma=None, labelDistribution = True)).reshape(-1)
            plt.figure()
            plt.hist(simDistArray[~np.isnan(simDistArray)])
            plt.title(figName)
            plt.savefig(figName)
    return            

#####

os.chdir('./..')

os.chdir('./percentileImages')

histCreator(metricLst, smallerLabelsList)    


################Generating Sim and Dist Matrices###############################
X_data = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1 , usecols = (range(11,76)))
Y_data = np.genfromtxt('./LIDC_REU2015.csv', delimiter = ',', skip_header = 1, usecols = (84,93,102,111)).astype(int)

X_data = np.genfromtxt('./naturalSceneFeatures.csv', delimiter = ',')
Y_data = np.genfromtxt('./naturalSceneLabels.csv', delimiter = ',')

X_data = np.genfromtxt('./SJAFeatures.csv', delimiter = ',')
Y_data = np.genfromtxt('./SJALabels.csv', delimiter = ',')



trainX, trainY, testX, testY = splitTrainTest(X_data,Y_data,0.7,99)

X,S,D,R = genSimDistRatioMats(data = trainX, targetArray = trainY, scale ='01',alpha = 0.5)

os.chdir('./..')

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
