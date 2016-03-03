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
    return np.mat(dataArr), np.mat(labelArr).T


def standardRegression(dataMat, labelMat):
    xTx = dataMat.T * dataMat
    if np.linalg.det(xTx) == 0:
        raise NameError('x is a singular matrix')
    theta = xTx.I * dataMat.T * labelMat
    return theta


def gradDescent(dataMat, labelMat, numIter=1000):
    alpha = 0.001
    m, n = np.shape(dataMat)
    theta = np.ones((n, 1))  # shape: n x 1
    for j in range(numIter):
        h = dataMat * theta
        error = h - labelMat  # shape: m x 1
        # shape: (n x m) * (m x 1)
        theta = theta - alpha * dataMat.T * error
    return theta


def stocGradDescent(dataMat, labelMat, numIter=1000):
    m, n = np.shape(dataMat)
    theta = np.ones((n, 1))  # shape: n x 1
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + i + j) + 0.0001
            randId = int(np.random.uniform(0, len(dataIndex)))
            h = dataMat[randId] * theta  # shape:(1 x n) * (n * 1)
            # shape: 1 x 1
            error = h - labelMat[randId]
            # shape:  (n x 1) = (n x 1) * (1 x 1)
            theta = theta - alpha * dataMat[randId].T * error
            del(dataIndex[randId])
    return theta


def testForSimppleRegression():
    dataMat, labelMat = loadData('ex0.txt')

    theta = standardRegression(dataMat, labelMat)
    print theta

    theta = gradDescent(dataMat, labelMat)
    print theta

    theta = stocGradDescent(dataMat, labelMat)
    print theta


def gradLWLR(inX, dataMat, labelMat, k=0.01, numIter=500):
    alpha = 0.001
    m, n = np.shape(dataMat)
    theta = np.ones((n, 1))
    W = np.eye(m)
    for i in range(m):
        diffMat = inX - dataMat[i, :]
        W[i, i] = np.exp((diffMat * diffMat.T) / (-2.0 * k ** 2))
    return

def testLWLR():
    dataMat, labelMat = loadData('ex0.txt')
    gradLWLR(dataMat[1], dataMat, labelMat)

if __name__ == '__main__':
    print 'start:'
    # testForSimppleRegression()
    testLWLR()




