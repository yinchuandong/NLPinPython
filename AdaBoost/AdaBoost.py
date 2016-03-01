# encoding:utf-8


import numpy as np


def loadSimpData():
    dataMat = np.mat([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    labelMat = np.mat([1.0, 1.0, -1.0, -1.0, 1.0]).T
    return dataMat, labelMat


def loadDataFromFile(filename):
    fr = open(filename)
    dataArr = []
    labelArr = []
    for line in fr.readlines():
        rowArr = []
        lineArr = line.strip().split('\t')
        for i in range(len(lineArr) - 1):
            rowArr.append(float(lineArr[i]))
        dataArr.append(rowArr)
        labelArr.append(float(lineArr[-1]))
    return np.mat(dataArr), np.mat(labelArr).T


def stumpClassify(dataMat, dim, threshVal, threshIneq):
    m = np.shape(dataMat)[0]
    retArr = np.ones((m, 1))  # predicted class vector
    if threshIneq == 'lt':
        retArr[dataMat[:, dim] <= threshVal] = -1.0
    else:
        retArr[dataMat[:, dim] > threshVal] = -1.0
    return retArr


def buildStump(dataMat, labelMat, D):
    m, n = np.shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat(np.zeros((m, 1)))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        # loop from min to max in current dimension
        for j in range(-1, int(numSteps) + 1):
            for threshIneq in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predictedVals = stumpClassify(dataMat, i,
                                              threshVal, threshIneq)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # calculate the total error by matrix multiplication
                weightedErr = D.T * errArr
                if weightedErr < minError:
                    minError = weightedErr
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['threshIneq'] = threshIneq
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataMat, labelMat, numIter=40):
    """
    " param dataMat: 
    " param labelMat: column vector
    """
    weakClassArr = []
    m = np.shape(dataMat)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIter):
        bestStump, error, classEst = buildStump(dataMat, labelMat, D)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # calculate the exponent of D
        expon = np.multiply(-1 * alpha * labelMat, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # calculate the aggregate error
        aggClassEst += alpha * classEst
        aggErrors = np.multiply(
            np.sign(aggClassEst) != labelMat, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print 'total error: ', errorRate
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(dataToClass, classifierArr):
    """
    " predict function
    """
    dataMat = np.mat(dataToClass)
    m = np.shape(dataMat)[0]
    aggClassEst = np.mat(np.zeros((m, 1)))
    for classifier in classifierArr:
        classEst = stumpClassify(
            dataMat, classifier['dim'],
            classifier['threshVal'], classifier['threshIneq'])
        aggClassEst += classifier['alpha'] * classEst
        # print aggClassEst
    return np.sign(aggClassEst)

if __name__ == '__main__':
    print 'start'
    # dataMat, labelMat = loadSimpData()
    # classifierArr, aggClassEst = adaBoostTrainDS(dataMat, labelMat)
    # result = adaClassify([0, 0], classifierArr)
    # result = adaClassify([[5, 5], [0, 0]], classifierArr)
    # print result

    trainMat, trainLabelMat = loadDataFromFile('horseColicTraining2.txt')
    testMat, testLabelMat = loadDataFromFile('horseColicTest2.txt')
    classifierArr, aggClassEst = adaBoostTrainDS(trainMat, trainLabelMat, 70)
    result = adaClassify(testMat, classifierArr)
    errorArr = np.mat(np.ones((67, 1)))
    errorNum = errorArr[result != testLabelMat].sum()
    print 'test error: ', errorNum / 67.00
    # print isinstance(1.0, int)
