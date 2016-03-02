# coding=utf-8


import numpy as np


def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def sigmoid(inX):
    return 1.0 / (1.0 + np.exp(-inX))


def gradAscent(dataMatIn, labelMatIn):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(labelMatIn).T
    m, n = np.shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMat.T * error
    return weights


def stocGradAscent(dataMatIn, labelMatIn, numIter=150):
    m, n = np.shape(dataMatIn)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4.0 / (1.0 + j + i) + 0.0001
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatIn[randIndex] * weights))
            error = labelMatIn[randIndex] - h
            weights = weights + alpha * error * dataMatIn[randIndex]
            del(dataIndex[randIndex])
    return weights


def classifyVector(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def horseColicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainSet = []
    trainLabel = []
    for line in frTrain.readlines():
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(curLine[i]))
        trainSet.append(lineArr)
        trainLabel.append(float(curLine[21]))

    trainWeights = stocGradAscent(np.array(trainSet), trainLabel, 1000)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(curLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != int(curLine[21]):
            errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print 'error rate is: %f' % errorRate
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += horseColicTest()
    print 'after %d iterations, the average error rate is: %f' % (numTests, errorSum / float(numTests))



def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()



if __name__ == '__main__':
    dataMat, labelMat = loadDataSet()
    # weights = gradAscent(dataMat, labelMat)
    # plotBestFit(weights.getA())
    # weights = stocGradAscent(np.array(dataMat), labelMat, 200)
    # weights = horseColicTest()
    # plotBestFit(weights)
    # print weights
    
    multiTest()







