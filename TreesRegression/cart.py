import numpy as np


def loadData(filename):
    dataArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        row = map(float, lineArr)
        dataArr.append(row)
    return np.mat(dataArr)


def binSplit(dataMat, featureId, value):
    idx0 = np.nonzero(dataMat[:, featureId] > value)[0]
    idx1 = np.nonzero(dataMat[:, featureId] <= value)[0]
    return dataMat[idx0], dataMat[idx1]


def regLeaf(dataMat):
    # caluate the mean value for each leaf
    return np.mean(dataMat[:, -1])


def regError(dataMat):
    # calculate the variance
    return np.var(dataMat[:, -1]) * np.shape(dataMat)[0]


def chooseBestFeature(dataMat, leafType=regLeaf, errType=regError, ops=(1, 4)):
    """
    Parameters
    ------------
    dataMat : numpy.matrix
    leafType : regLeaf(dataMat), optional
    errType : regError(dataMat), optional
    ops: (tolS, tolN), optional
        tolS is a tolerance on the error reduction
        tolN is the minimum data instances to include in a split
    """ 
    tolS = ops[0]
    tolN = ops[1]
    # if all values are the same, quit and return the values
    if len(set(dataMat[:, -1].T.tolist())) == 1:
        return None, leafType(dataMat)
    m, n = np.shape(dataMat)
    
    return



if __name__ == '__main__':
    print 'start'
    dataMat = loadData('ex0.txt')
    subMat1, subMat2 = binSplit(dataMat, 2, 2.4)
    np.var















