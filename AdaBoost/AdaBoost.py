# encoding:utf-8


import numpy as np


def loadSimpData():
    datMat = np.mat([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMat, dim, threshVal, threshIneq):
    retArr = np.ones((np.shape(dataMat)[0], 1))
    if threshIneq == 'lt':
        retArr[dataMat[:, dim] <= threshVal] = -1.0
    else:
        retArr[dataMat[:, dim] > threshVal] = -1.0
    return retArr


def buildStump(dataArr, labelArr, D):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr)
    m, n = np.shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat((m, 1))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMax) / numSteps
        


def adaBoostTrainDS(dataArr, labelArr, numIter=40):


def adaClassify(datToClass, classifierArr):

if __name__ == '__main__':
    print 'start'
