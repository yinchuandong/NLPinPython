import numpy as np


def loadData(filename):
    dataArr = []
    labelArr = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        row = []
        for i in range(len(lineArr) - 1):
            row.append(float(lineArr[i]))
        dataArr.append(row)
        labelArr.append(float(lineArr[-1]))
    return dataArr, labelArr


def standardRegression(dataMat, labelMat):
    xTx = dataMat.T * dataMat
    if np.linalg.det(xTx) == 0:
        raise NameError('x is a singular matrix')
    W = xTx.I * dataMat.T * labelMat
    return W


def gradDescent(dataMat, labelMat, numIter=1000):
    alpha = 0.001
    m, n = np.shape(dataMat)
    weights = np.ones((n, 1))  # shape: n x 1
    for j in range(numIter):
        h = dataMat * weights
        error = h - labelMat  # shape: m x 1
        # shape: (n x m) * (m x 1)
        weights = weights - alpha * dataMat.T * error
    return weights


def stocGradDescent(dataMat, labelMat, numIter=1000):
    m, n = np.shape(dataMat)
    weights = np.ones(n)  # shape: 1 x n
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + i + j) + 0.0001
            randId = int(np.random.uniform(0, len(dataIndex)))
            h = sum(dataMat[randId] * weights)  # shape:(1 x n) * (n * 1)
            error = h - labelMat[randId]  # shape: 1 x 1
            # shape: (1 x 1) * (1 x n) = 1 x n
            weights = weights - alpha * error * dataMat[randId]
            del(dataIndex[randId])

    return weights

if __name__ == '__main__':
    print 'start:'
    dataArr, labelArr = loadData('ex0.txt')
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    W = standardRegression(dataMat, labelMat)
    print W

    W = stocGradDescent(np.array(dataArr), labelArr)
    print W

    # W = gradDescent(dataMat, labelMat)
    # print W
    # print dataMat
    # print labelMat
