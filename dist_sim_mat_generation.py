# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 21:18:48 2016

@author: syarlag1
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('C:/Users/syarlag1/Desktop/Label-Distribution-Metric-Learning/data')
#os.chdir('/Users/Sriram/Desktop/DePaul/Label-Distribution-Metric-Learning/data')

###############FUNCTIONS############################

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
    


def genSimDistMat(measure, labels, sigma=None, labelDistribution = True): 
    if type(labels) == str: Y = globals()[labels]
    if labelDistribution: pass
    else: Y = createDistributionLabels(Y)     
    S = np.zeros(shape=[Y.shape[0], Y.shape[0]])
    if measure == 'gaussian': return gaussSimMatrix(labels, sigma)[0]
    for i in range(S.shape[0]):
        for j in range(S.shape[0]):
            if measure == 'cosine': S[i,j] = np.dot(Y[i],Y[j])/(np.linalg.norm(Y[i])*np.linalg.norm(Y[j]))
            if measure == 'fidelity': S[i,j] = np.sum(np.sqrt(np.multiply(Y[i],Y[j])))
            if measure == 'intersection': S[i,j] =  np.sum(np.minimum(Y[i],Y[j]))   
            if measure == 'euclidean': S[i,j] = np.sqrt(np.sum(np.dot((Y[i]-Y[j]),(Y[i]-Y[j]))))
            if measure == 'sorensen': S[i,j] = np.sum(np.abs(Y[i]-Y[j]))/(np.sum(Y[i]+Y[j])) 
            if measure == 'squaredChiSq': S[i,j] = np.sum(np.square(Y[i]-Y[j])/(Y[i]+Y[j]))
            if measure == 'chebyshev': S[i,j] = np.max(np.abs(Y[i] - Y[j]))            
            if measure == 'clark': S[i,j] =  np.sqrt(np.sum(np.square(Y[i]-Y[j])/np.square((Y[i]+Y[j]))))
            if measure == 'canberra': S[i,j] = np.sum(np.abs(Y[i]-Y[j])/((Y[i]+Y[j])))
           #if measure == 'gaussian': S[i,j] = np.exp(-(np.sqrt(np.sum(np.dot((Y[i]-Y[j]),(Y[i]-Y[j])))))**2/(2*sigma**2))
            if measure == 'KL':
                tempSum = 0
                for k in range(len(Y[i])):
                    tempSum += Y[i,k]*(np.log((Y[i,k] if Y[i,k]>0 else 0.01)/(Y[j,k] if Y[j,k]>0 else 0.01)))
                S[i,j] = tempSum
    return S

def gaussSimMatrix(labels, sigma=None):
    Y=labels    
    euclideanSimMat =  genSimDistMat('euclidean',Y)
    if sigma is None: sigma = np.nanstd(euclideanSimMat)
    return np.exp(-euclideanSimMat/(2*(sigma)**2)), sigma  
        

def metricStats(metricList, labels):
    mean = []; std = []; maxLst = []; minLst = []; nanCount = []
    for metric in metricList: #Nan values are ignored when calculating the performance
        S = genSimDistMat(metric, labels)
        nanCount.append(np.sum(np.isnan(S)))
        mean.append(np.nanmean(S))       
        std.append(np.nanstd(S))
        maxLst.append(np.nanmax(S))
        minLst.append(np.nanmin(S))
    combinedList = zip(metricList, mean, std, maxLst, minLst, nanCount)
    colNames = ['Metric', 'Mean', 'StdDev', 'MaxValue', 'MinValue', 'NanCount']
    return pd.DataFrame(combinedList, columns = colNames)
    

def metricStatsforLabelList(metricList, labelsList):
    resultDict = {}
    for labels in labelsList:
        result = metricStats(metricList, labels)
        resultDict[labels] = result
        print labels
        print result
        print '\n'
    return resultDict

def histCreator(metricList, labelsList):
    for metric in metricList:
        for labels in labelsList:
            figName = metric + '-' + labels + '.png'
            simDistArray = np.asarray(genSimDistMat(metric, labels, sigma=None, labelDistribution = True)).reshape(-1)
            plt.figure()
            plt.hist(simDistArray)
            plt.title(figName)
            plt.savefig(figName)
    return            
        
        

#################SCRIPT TO CALC THE MATRICES##########

metrics = ['cosine', 'fidelity', 'intersection', 'euclidean', 'sorensen', 'squaredChiSq',\
               'chebyshev', 'clark', 'canberra','KL', 'gaussian']

#trial = np.array([[0.1, 0.2, 0.3, 0.4],[0.5, 0.5, 0, 0], [0.1, 0.3, 0.6, 0], [0.2, 0.4, 0.1, 0.3]])
#metricStats(metrics, trial)

labelsList = []
for fileName in os.listdir('./'):
    if 'Label' in fileName:
        locals()['{0}'.format(fileName)] = np.genfromtxt(fileName, delimiter=',') #This is probably not safe to use
        labelsList.append(fileName)
labelsList

smallerLabelsList = ['SJALabels.csv','naturalSceneLabels.csv', 'YeastSPOEMLabels.csv', 'YeastHeatLabels.csv', 'YeastSPOEMLabels.csv' ]

results = metricStatsforLabelList(metrics, smallerLabelsList)

for filename in smallerLabelsList:
    S, EuclStddev =  gaussSimMatrix(filename, sigma=None)
    print 'sigma = ', EuclStddev, \
    'mean =', np.nanmean(S), 'stddev =', np.nanstd(S),\
    'max =', np.nanmax(S), 'min =', np.nanmin(S)

os.chdir('./..')
os.mkdir('./images')
os.chdir('./images')
histCreator(metrics, smallerLabelsList)  
    
plt.hist(S, bins = 20) ##IMP: all the values are clustered around 0...



