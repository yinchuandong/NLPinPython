# encoding:utf-8
__author__ = 'wangjiewen'

import json
from math import log


def loadFile(filename):
    """
    加载数据文件
    :param filename: 文件路径
    :return: void
    """
    fr = open(filename)
    lines = fr.readlines()

    dataSet = []
    for lineStr in lines:
        lineArr = lineStr.strip().split(' ')
        dataSet.append(lineArr)
    return dataSet


def calcEntropy(dataSet):
    nums = len(dataSet)
    labelsCount = {}
    for featVec in dataSet:
        label = featVec[-1]
        if label not in labelsCount:
            labelsCount[label] = 1
        else:
            labelsCount[label] += 1
    entropy = 0.0
    for label in labelsCount:
        prob = float(labelsCount[label]) / nums
        entropy -= prob * log(prob)
    return entropy


def splitDataSet(dataSet, col):
    result = {}
    for featVec in dataSet:
        key = featVec[col]
        tmpArr = featVec[:col]
        tmpArr.extend(featVec[col + 1:])
        if key not in result:
            result[key] = []
            result[key].append(tmpArr)
        else:
            result[key].append(tmpArr)
    return result


def selectMaxGainCol(dataSet):
    """
    选择信息最大信息增益的列标号
    :param dataSet:
    :return:
    """
    numsOfCol = len(dataSet[0]) - 1  # 数据集的列数，最后一列为类标号
    numsOfData = len(dataSet)  # 数据集的条目数量
    maxGain = 0.0  # 最大的信息增益
    maxFeatCol = -1  # 最大信息增益对应的列
    # Entropy(S)
    entropyS = calcEntropy(dataSet)
    for col in range(0, numsOfCol):
        featDict = splitDataSet(dataSet, col)
        tmpGain = entropyS
        for key in featDict.keys():
            featArr = featDict[key]
            entropyFeat = calcEntropy(featArr)
            delta = (len(featArr) / float(numsOfData)) * entropyFeat
            tmpGain -= delta
        # print "Gain(", labels[col], ") =", tmpGain
        if tmpGain > maxGain:
            maxGain = tmpGain
            maxFeatCol = col
    return maxFeatCol


def createTree(dataSet, labels):
    classList = [featVec[-1] for featVec in dataSet]
    if len(set(classList)) == 1:
        return classList[0]
    bestCol = selectMaxGainCol(dataSet)
    bestColLabel = labels[bestCol]
    tree = {}
    tree[bestColLabel] = {}
    labels = labels[:]  # copy in case of bad reference
    del(labels[bestCol])  # del after spliting on it
    featVals = [featVec[bestCol] for featVec in dataSet]
    uniqueVals = set(featVals)
    subData = splitDataSet(dataSet, bestCol)
    for val in uniqueVals:
        subtree = createTree(subData[val], labels[:])
        tree[bestColLabel][val] = subtree
    return tree


def classify(tree, featLabels, testVec):
    firstKey = tree.keys()[0]
    secondDict = tree[firstKey]
    featIndex = featLabels.index(firstKey)
    key = testVec[featIndex]
    featValue = secondDict[key]
    if isinstance(featValue, dict):
        classLabel = classify(featValue, featLabels, testVec)
    else:
        classLabel = featValue
    return classLabel


if __name__ == '__main__':

    dataSet = loadFile('trainset.txt')
    featLabels = ['Outlook', 'Temperature', 'Humidity', 'Wind']
    testVec = "Rain Mild High Weak".split(" ")
    testVec2 = "Overcast Mild High Weak".split(" ")
    tree = createTree(dataSet, featLabels)
    print json.dumps(tree, indent=4)
    result = classify(tree, featLabels, testVec)
    print result
