# coding=utf-8

import numpy as np

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def trainNB(trainMat, labelMat):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    # in labelMat, 1 stand for abusive; 0 for normal
    pClass1 = sum(labelMat) / float(numTrainDocs)
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if labelMat[i] == 1:
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    # regarding the accuracy of float in python
    # change it to log
    p0Vec = np.log(p0Num / p0Denom)
    p1Vec = np.log(p1Num / p1Denom)
    return p0Vec, p1Vec, pClass1


def classifyBN(vec2Classify, p0Vec, p1Vec, pClass1):
    p0 = sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    if p0 > p1:
        return 0
    else:
        return 1


def textParse(bigString):
    """
    " filter too short words
    " input is big string, output is a list
    """
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    docList = []
    classList = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(1)

        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = range(50)
    testSet = []
    for i in range(10):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainMat = []
    labelMat = []
    for docId in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docId]))
        labelMat.append(classList[docId])

    p0Vec, p1Vec, pSpam = trainNB(np.array(trainMat), np.array(labelMat))
    errorCount = 0
    for docId in testSet:
        wordVec = bagOfWords2VecMN(vocabList, docList[docId])
        if classifyBN(wordVec, p0Vec, p1Vec, pSpam) != classList[docId]:
            errorCount += 1
            print "classfiy wrongly", docList[docId]

    print 'the error rate is: ', float(errorCount) / len(testSet)









if __name__ == '__main__':
    print 'begin'
    spamTest()
    print 'end'


