# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 21:18:48 2016

@author: syarlag1
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###############FUNCTIONS############################

# return distribution labels for the LIDC-like datasets
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
    

# returns matrix for various sim and dist metrics
def genSimDistMat(measure, labels, sigma=None, labelDistribution = True, percentile=True): 
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

# returns a Gaussian similarity matrix
def gaussSimMatrix(labels, sigma=None):
    Y=labels    
    euclideanSimMat =  genSimDistMat('euclidean',Y)
    if sigma is None: sigma = np.nanstd(euclideanSimMat)
    return np.exp(-euclideanSimMat**2/(2*(sigma)**2)), sigma  
        
# returns important stats for the labels
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
    
# return a percentile matrix for the inputted matrix
def convertMatToPercentile(S):
    originalShape = S.shape
    arr = S.flatten()
    sortedUniqueArr = np.unique(np.sort(arr))
    percentileArr = np.linspace(0.01,1,len(sortedUniqueArr))
    percentileValueDict = {x:percentileArr[i] for i,x in enumerate(sortedUniqueArr)}
    percentileArr = np.array([percentileValueDict[x] for x in arr])
    
    return np.reshape(percentileArr, originalShape)
    
   
# returns stats for each element in the list
def metricStatsforLabelList(metricList, labelsList):
    resultDict = {}
    for labels in labelsList:
        result = metricStats(metricList, labels)
        resultDict[labels] = result
        print labels
        print result
        print '\n'
    return resultDict

# prints histograms for the inputted matrices
def histCreator(metricList, labelsList):
    for metric in metricList:
        for labels in labelsList:
            figName = metric + '-' + labels + '.png'
            simDistArray = np.asarray(genSimDistMat(metric, labels, sigma=None, labelDistribution = True)).reshape(-1)
            plt.figure()
            plt.hist(simDistArray[~np.isnan(simDistArray)])
            plt.title(figName)
            plt.savefig(figName)
    return            
        
        


