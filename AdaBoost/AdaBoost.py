#encoding:utf-8


import numpy as np


def loadSimpData():
    datMat = np.mat([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels



def buildStump(dataArr, labelArr, D):


def adaBoostTrainDS(dataArr, labelArr, numIter=40):


def adaClassify(datToClass, classifierArr):

if __name__ == '__main__':
    
