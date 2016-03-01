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

    return

if __name__ == '__main__':
    print 'start:'
    dataMat, labelMat = loadData('ex0.txt')
    
