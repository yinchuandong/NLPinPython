#encoding:utf-8
__author__ = 'wangjiewen'


import numpy as np

def loadSimpData():
    datMat = np.mat([[ 1. ,  2.1],
        [ 2. ,  1.1],
        [ 1.3,  1. ],
        [ 1. ,  1. ],
        [ 2. ,  1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat,classLabels

def test():
    dataMat, classLabels = loadSimpData()
    m, n = np.shape(dataMat)
    aggClassEst = np.mat([1, 1, -1.0, 1.0, -1.0]).T

    print np.sign(aggClassEst) != np.mat(classLabels).T
    aggError = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T,
                           np.ones((m, 1)))
    errorRate = aggError.sum() / m
    print aggError
    print errorRate
    return


test()