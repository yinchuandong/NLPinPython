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
    if len(set(dataMat[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataMat)
    m, n = np.shape(dataMat)
    S = errType(dataMat)  # sum of dataMat error
    bestS = np.inf
    bestIndex = 0
    bestValue = 0
    for featIndex in range(n - 1):
        for featValue in set(dataMat[:, featIndex].T.tolist()[0]):
            mat0, mat1 = binSplit(dataMat, featIndex, featValue)
            if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestS = newS
                bestIndex = featIndex
                bestValue = featValue

    # if the decrease (S - newS) is less than threshold tolS,
    # then stop spliting
    if (S - newS) < tolS:
        return None, leafType(dataMat)

    mat0, mat1 = binSplit(dataMat, bestIndex, bestValue)
    if np.shape(mat0)[0] < tolN or np.shape(mat1)[0] < tolN:
        return None, leafType(dataMat)
    return bestIndex, bestValue


def createTree(dataMat, leafType=regLeaf, errType=regError, ops=(1, 4)):
    featIndex, featValue = chooseBestFeature(dataMat, leafType, errType, ops)
    if featIndex is None:
        return featValue

    retTree = {}
    retTree['featIndex'] = featIndex
    retTree['featValue'] = featValue
    nodeStack = []
    nodeStack.append(retTree)
    while len(nodeStack) != 0:
        curNode = nodeStack.pop()
        if curNode['featIndex'] is None:
            continue
        leftMat, rightMat = binSplit(dataMat, curNode['featIndex'], curNode['featValue'])
        leftIndex, leftValue = chooseBestFeature(leftMat, leafType, errType, ops)
        rightIndex, rightValue = chooseBestFeature(rightMat, leafType, errType, ops)
        leftTree = {}
        leftTree['featIndex'] = leftIndex
        leftTree['featValue'] = leftValue
        curNode['left'] = leftTree
        rightTree = {}
        rightTree['featIndex'] = rightIndex
        rightTree['featValue'] = rightValue
        curNode['right'] = rightTree
        nodeStack.append(rightTree)
        nodeStack.append(leftTree)

    print retTree
    return retTree


if __name__ == '__main__':
    print 'start'
    dataMat = loadData('ex0.txt')
    # subMat1, subMat2 = binSplit(dataMat, 2, 2.4)
    createTree(dataMat)




