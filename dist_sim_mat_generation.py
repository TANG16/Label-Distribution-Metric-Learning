# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 21:18:48 2016

@author: syarlag1
"""

import os
import numpy as np
import pandas as pd

os.chdir('C:/Users/syarlag1/Desktop/Label-Distribution-Metric-Learning/data')


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
    


def genSimDistMat(measure, labels, labelDistribution = True): 
    if type(labels) == str: Y = globals()[labels]
    if labelDistribution: pass
    else: Y = createDistributionLabels(Y)     
    S = np.zeros(shape=[Y.shape[0], Y.shape[0]])
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
            if measure == 'KL':
                tempSum = 0
                for k in range(len(Y[i])):
                    tempSum += Y[i,k]*np.log(Y[i,k]/(Y[j,k]))
                S[i,j] = tempSum
    return S
    

def metricStats(metricList, labels):
    mean = []; std = []; maxLst = []; minLst = []
    for metric in metricList: #Nan values are ignored when calculating the performance
        S = genSimDistMat(metric, labels)
        mean.append(np.nanmean(S))       
        std.append(np.nanstd(S))
        maxLst.append(np.nanmax(S))
        minLst.append(np.nanmin(S))
    combinedList = zip(metricList, mean, std, maxLst, minLst)
    colNames = ['Metric', 'Mean', 'StdDev', 'MaxValue', 'MinValue']
    return pd.DataFrame(combinedList, columns = colNames)
    

def metricStatsforLabelList(metricList, labelsList):
    resultList = []
    for labels in labelsList:
        result = metricStats(metricList, labels)
        resultList.append(result)
        print result
    return resultList
    

#################SCRIPT TO CALC THE MATRICES##########

metrics = ['cosine', 'fidelity', 'intersection', 'euclidean', 'sorensen', 'squaredChiSq',\
               'chebyshev', 'clark', 'canberra','KL']

#trial = np.array([[0.1, 0.2, 0.3, 0.4],[0.5, 0.5, 0, 0], [0.1, 0.3, 0.6, 0], [0.2, 0.4, 0.1, 0.3]])
#metricStats(metrics, trial)

labelsList = []
for fileName in os.listdir('./'):
    locals()['{0}'.format(fileName)] = np.genfromtxt(fileName, delimiter=',') #This is probably not safe to use
    labelsList.append(fileName)

results = metricStatsforLabelList(metrics, labelsList)




