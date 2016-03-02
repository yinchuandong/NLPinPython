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
    W = xTx.I * dataMat.T * labelMat
    return W


def gradientDescent(dataMat, labelMat):

    return

if __name__ == '__main__':
    print 'start:'
    dataMat, labelMat = loadData('ex0.txt')
    W = standardRegression(dataMat, labelMat)
    print W
    # print dataMat
    # print labelMat
