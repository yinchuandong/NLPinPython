# encoding:utf-8


import numpy as np


def loadSimpData():
    dataMat = np.mat([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    labelMat = np.mat([1.0, 1.0, -1.0, -1.0, 1.0])
    return dataMat, labelMat


def stumpClassify(dataMat, dim, threshVal, threshIneq):
    m = np.shape(dataMat)[0]
    retArr = np.ones((m, 1))  # predicted class vector
    if threshIneq == 'lt':
        retArr[dataMat[:, dim] <= threshVal] = -1.0
    else:
        retArr[dataMat[:, dim] > threshVal] = -1.0
    return retArr


def buildStump(dataArr, labelArr, D):
    dataMat = np.mat(dataArr)
    labelMat = np.mat(labelArr).T
    m, n = np.shape(dataMat)
    numSteps = 10.0
    bestStump = {}
    bestClassEst = np.mat((m, 1))
    minError = np.inf
    for i in range(n):
        rangeMin = dataMat[:, i].min()
        rangeMax = dataMat[:, i].max()
        stepSize = (rangeMax - rangeMax) / numSteps
        # loop from min to max in current dimension
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predictedVals = stumpClassify(dataMat, i,
                                              threshVal, inequal)
                errArr = np.mat(np.ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                # calculate the total error by matrix multiplication
                weightedErr = D.T * errArr
                if weightedErr < minError:
                    minError = weightedErr
                    bestClassEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['threshIneq'] = inequal
    return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataArr, labelArr, numIter=40):
    weakClassArr = []
    m = np.shape(dataArr)[0]
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIter):
        bestStump, error, classEst = buildStump(dataArr, labelArr, D)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # calculate the exponent of D
        expon = np.multiply(-1 * alpha * np.mat(labelArr).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()

        # calculate the aggregate error
        aggClassEst += alpha * classEst
        aggErrors = np.multiply(
            np.sign(aggClassEst) != np.mat(labelArr).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print 'total error: ', errorRate
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(dataToClass, classifierArr):
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
    # a = np.mat([1, 1, 1])
    # b = np.mat([1, 1, 0])
    # print np.multiply(a == b, a.T)
    print 'start'
    dataArr, labelArr = loadSimpData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataArr, labelArr)
    # result = adaClassify([0, 0], weakClassArr)
    result = adaClassify([[5, 5],[0,0]], weakClassArr)
    print result
